!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f00(x)
  implicit none
  logical :: x
  !ERROR: Must be a constant value
  !$omp task replayable(x)
  !$omp end task
end

subroutine f01
  !ERROR: Must have LOGICAL type, but is INTEGER(4)
  !$omp task replayable(7)
  !$omp end task
end

subroutine f02
  !No diagnostic expected
  !$omp task replayable
  !$omp end task
end

