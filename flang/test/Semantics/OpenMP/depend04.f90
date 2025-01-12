!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50

subroutine f00
  integer :: x
  common /cc/ x
!ERROR: Common block name ('cc') cannot appear in a DEPEND clause
  !$omp task depend(in: /cc/)
  x = 0
  !$omp end task
end
