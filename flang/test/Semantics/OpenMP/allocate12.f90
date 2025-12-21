!RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=51

subroutine f00
  integer, allocatable :: x
  continue
  !ERROR: An executable ALLOCATE directive must be associated with an ALLOCATE statement
  !$omp allocate(x)
end

subroutine f01
  integer, allocatable :: x
  continue
  !$omp allocate(x)
  !ERROR: The statement associated with executable ALLOCATE directive must be an ALLOCATE statement
  continue
end
