!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=61

subroutine f00
  !ERROR: The argument to the dims-modifier should be positive
  !$omp teams num_teams(dims(0): 1)
  !$omp end teams
end

subroutine f01(x)
  integer :: x
  !ERROR: Must be a constant value
  !$omp teams num_teams(dims(x): 1)
  !$omp end teams
end

subroutine f02
  !ERROR: The dims-modifier specifies 2 dimensions but 3 values were provided
  !$omp teams num_teams(dims(2): 1, 2, 3)
  !$omp end teams
end

subroutine f03
  !ERROR: The parameter of the NUM_TEAMS clause must be a positive integer expression
  !$omp teams num_teams(dims(2): -2, 3)
  !$omp end teams
end
