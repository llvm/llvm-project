! RUN: %python %S/test_errors.py %s %flang_fc1
! Regression test for crash
subroutine sub(xx)
  type(*) :: xx
  type ty
  end type
  type(ty) obj
  !ERROR: TYPE(*) dummy argument may only be used as an actual argument
  obj = ty(xx)
end
