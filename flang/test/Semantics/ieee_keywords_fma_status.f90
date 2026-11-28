! RUN: %python %S/test_errors.py %s %flang_fc1
! IEEE_FMA has the standard dummy argument names A, B, C (F2023 17.11.4).
! IEEE_GET_STATUS and IEEE_SET_STATUS, from IEEE_EXCEPTIONS, have STATUS_VALUE.
use ieee_arithmetic, only : ieee_fma, ieee_get_status, ieee_set_status, &
  ieee_status_type
implicit none
real :: x = 1.0, y = 2.0, z = 3.0
type(ieee_status_type) :: s
print *, ieee_fma(x, y, z)
print *, ieee_fma(A = x, B = y, C = z)
print *, ieee_fma(C = z, A = x, B = y)
!ERROR: unknown keyword argument to intrinsic '__builtin_fma'
print *, ieee_fma(F1 = x, F2 = y, F3 = z)
!ERROR: unknown keyword argument to intrinsic '__builtin_fma'
print *, ieee_fma(X = x, Y = y, Z = z)
call ieee_get_status(s)
call ieee_get_status(STATUS_VALUE = s)
!ERROR: No specific subroutine of generic 'ieee_get_status' matches the actual arguments
call ieee_get_status(STATUS = s)
call ieee_set_status(s)
call ieee_set_status(STATUS_VALUE = s)
!ERROR: No specific subroutine of generic 'ieee_set_status' matches the actual arguments
call ieee_set_status(STATUS = s)
end
