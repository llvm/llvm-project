! RUN: %python %S/test_errors.py %s %flang_fc1
! Check that EXTERNAL/INTRINSIC and TARGET attributes conflict

subroutine test1()
  !ERROR: 'f1' may not have both the EXTERNAL and TARGET attributes
  real, target, external :: f1
  !ERROR: 'f2' may not have both the EXTERNAL and TARGET attributes
  real, external, target :: f2
  ! These are fine individually
  real, external :: f3
  real, target :: x1
end subroutine

subroutine test2()
  !ERROR: 'abs' may not have both the INTRINSIC and TARGET attributes
  integer, target, intrinsic :: abs
end subroutine
