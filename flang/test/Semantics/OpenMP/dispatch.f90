! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

subroutine sb1
  integer :: r
  r = 1
  !$omp dispatch nowait
!ERROR: The body of the DISPATCH construct should be a function or a subroutine call
  print *,r
end subroutine

subroutine sb2
!ERROR: The DISPATCH construct should contain a single function or subroutine call
  !$omp dispatch
  !$omp end dispatch
end subroutine

subroutine sb3
!ERROR: The DISPATCH construct should contain a single function or subroutine call
  !$omp dispatch
end subroutine
