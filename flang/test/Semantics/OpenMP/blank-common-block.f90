!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

module m
  integer :: a
  common // a
  !ERROR: Blank common blocks are not allowed as directive or clause arguments
  !$omp declare_target(//)
  !ERROR: Blank common blocks are not allowed as directive or clause arguments
  !$omp threadprivate(//)
end

subroutine f00
  integer :: a
  common // a
  !ERROR: Blank common blocks are not allowed as directive or clause arguments
  !$omp parallel shared(//)
  !$omp end parallel
end
