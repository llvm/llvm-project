! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Check acc routine in the top level.

subroutine sub1(a, n)
  integer :: n
  real :: a(n)
end subroutine sub1

!$acc routine(sub1)

!dir$ value=1
program test
  integer, parameter :: N = 10
  real :: a(N)
  call sub1(a, N)
end program
