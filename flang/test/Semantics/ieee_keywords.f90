! RUN: %python %S/test_errors.py %s %flang_fc1
! The mandatory dummy argument of the IEEE_ARITHMETIC classification inquiry
! functions is named X in the standard, not A.
use ieee_arithmetic, only : ieee_is_nan, ieee_is_negative, ieee_is_normal
implicit none
real :: v = 1.0
print *, ieee_is_nan(v)
print *, ieee_is_nan(X = v)
print *, ieee_is_negative(X = v)
print *, ieee_is_normal(X = v)
!ERROR: unknown keyword argument to intrinsic '__builtin_ieee_is_nan'
print *, ieee_is_nan(A = v)
!ERROR: unknown keyword argument to intrinsic '__builtin_ieee_is_negative'
print *, ieee_is_negative(A = v)
!ERROR: unknown keyword argument to intrinsic '__builtin_ieee_is_normal'
print *, ieee_is_normal(A = v)
end
