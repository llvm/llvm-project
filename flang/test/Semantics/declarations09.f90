! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic

subroutine bug149771

  !ERROR: 'x1' may not have both the EXTERNAL and INTRINSIC attributes
  integer, external, intrinsic :: x1

  !ERROR: 'x2' may not have both the EXTERNAL and PARAMETER attributes
  integer, external, parameter :: x2

end subroutine bug149771
