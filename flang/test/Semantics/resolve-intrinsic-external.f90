! RUN: %python %S/test_errors.py %s %flang_fc1

real function eval(x, n)
  real :: x
  integer :: n
  !ERROR: Attributes 'EXTERNAL' and 'INTRINSIC' conflict with each other
  real, external, intrinsic :: exp
  eval = 1.0
end function eval
