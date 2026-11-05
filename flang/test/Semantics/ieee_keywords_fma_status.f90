! RUN: %python %S/test_errors.py %s %flang_fc1
! The IEEE_ARITHMETIC procedures IEEE_FMA and IEEE_GET_STATUS/IEEE_SET_STATUS
! have the standard dummy argument names X, Y, Z and STATUS_VALUE.
use ieee_arithmetic, only : ieee_fma, ieee_get_status, ieee_set_status, &
  ieee_status_type
implicit none
real :: x = 1.0, y = 2.0, z = 3.0
type(ieee_status_type) :: s
print *, ieee_fma(x, y, z)
print *, ieee_fma(X = x, Y = y, Z = z)
print *, ieee_fma(Z = z, X = x, Y = y)
!ERROR: unknown keyword argument to intrinsic '__builtin_fma'
print *, ieee_fma(F1 = x, F2 = y, F3 = z)
call ieee_get_status(s)
call ieee_get_status(STATUS_VALUE = s)
!ERROR: No specific subroutine of generic 'ieee_get_status' matches the actual arguments
call ieee_get_status(STATUS = s)
call ieee_set_status(s)
call ieee_set_status(STATUS_VALUE = s)
!ERROR: No specific subroutine of generic 'ieee_set_status' matches the actual arguments
call ieee_set_status(STATUS = s)
end
