!RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=51

subroutine f00
  integer, allocatable :: x, y

  continue
  !$omp allocate
  !ERROR: If multiple directives are present in an executable ALLOCATE directive, at most one of them may specify no list items
  !$omp allocate
  allocate(x, y)
end
