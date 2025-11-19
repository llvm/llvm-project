!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine f00(y)
  integer :: x(10), y
  !ERROR: Must be a constant value
  !$omp allocate(x) align(y)
end

subroutine f01()
  integer :: x(10)
  !ERROR: The alignment should be a power of 2
  !$omp allocate(x) align(7)
end

subroutine f02()
  integer :: x(10)
  !ERROR: The alignment should be positive
  !$omp allocate(x) align(-8)
end

