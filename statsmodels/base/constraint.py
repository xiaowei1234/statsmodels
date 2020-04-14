import cvxpy as cp

class ConstraintProjector():
    """
    Find the point closest (L2 distance) to `goal` which satisfies the following constraints:
    1. x_min <= x <= x_max
    2. A@x <= b
    This class defines an optimization problem with fixed constraints, which can be re-solved
        repeatedly for different starting points. This should speed up runtime compared to
        re-creating the optimization problem from scratch for every call.
    A constraint feasibility check is performed at initialization. 
    
    Inputs:
        x_min (array-like): Minimum values for individual components of x. 1d array of size N.
        x_max (array-like): Maximum values for individual components of x. 1d array of size N.
        A (array-like): Constraint matrix. 2d array of size (K,N) where K is the number of constraints.
        b (array-like): Constraint right-hand side vector. 1d array of size K.
    """
    
    def __init__(self, x_min, x_max, A, b):
        # make sure that all inputs have appropriate dimensions
        if len(x_min) != len(x_max):
            raise ValueError(f'len(x_min)={len(x_min)}!={len(x_max)}=len(x_max)')
        if A.shape[1] == len(x_min):
            raise ValueError(f'A.shape[1]={A.shape[1]} incompatible with len(x_min)={len(x_min)}')
        if len(b) != A.shape[0]:
            raise ValueError(f'len(b)={len(b)} incompatible with A.shape[0]={A.shape[0]}')
        
        n_dim = len(x_min)
        self.x_min = x_min
        self.x_max = x_max
        self.A = A
        self.b = b
        self.x = cp.Variable(n_dim)
        self.goal = cp.Parameter(n_dim)
        # constrains: min, max, linear inequality
        constraints = [self.x >= x_min, self.x <= x_max, A@self.x <= b]
        cost = cp.sum_squares(self.x - self.goal)
        self.prob = cp.Problem(cp.Minimize(cost), constraints)
        # check feasibility
        self.goal.value = x_min
        self.prob.solve()
        if self.prob.status == 'infeasible':
            raise ValueError("The constraints are infeasible")
        
    def project(self, goal):
        """
        Project a specific point.

        Inputs:
            goal (array-like): Starting point which we need to project. 1d array of size N.
        Outputs:
            x (np.array): Result of the projection.
        """
        self.goal.value = goal
        self.x.value = goal # warm start from this point
        self.prob.solve(wamr_start=True)
        if self.prob.status != 'optimal':
            raise ValueError(f"Failed to solve optimization problem, status={self.prob.status}")
        return self.x.value
    