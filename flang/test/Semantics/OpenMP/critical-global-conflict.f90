! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -Werror

subroutine g
end

subroutine f(x)
  implicit none
  integer :: x

!ERROR: CRITICAL construct name 'f' conflicts with a previous declaration
  !$omp critical(f)
  x = 0
!ERROR: CRITICAL construct name 'f' conflicts with a previous declaration
  !$omp end critical(f)
end
