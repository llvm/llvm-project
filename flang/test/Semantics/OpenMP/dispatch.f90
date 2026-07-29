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

subroutine sb4
  logical :: c
  integer :: r
  ! The novariants clause is accepted; the body validation still applies.
  !$omp dispatch novariants(c)
!ERROR: The body of the DISPATCH construct should be a function or a subroutine call
  print *, r
end subroutine

subroutine sb5
  logical :: a, b
  ! novariants has the `unique` property (OpenMP 5.2, 7.6.1).
!ERROR: At most one NOVARIANTS clause can appear on DISPATCH directive
  !$omp dispatch novariants(a) novariants(b)
  call foo()
end subroutine

subroutine sb6
  integer :: i
  ! novariants requires an expression of logical type (OpenMP 5.2, 7.6.1).
!ERROR: Must have LOGICAL type, but is INTEGER(4)
  !$omp dispatch novariants(i)
  call foo()
end subroutine
