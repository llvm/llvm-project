! RUN: %python %S/test_errors.py %s %flang_fc1
use ieee_arithmetic, only: ieee_int, ieee_real, ieee_up
implicit none
print *, ieee_int(1.5, ieee_up)
print *, ieee_int(1.5, ieee_up, 4)
!ERROR: 'kind=' argument must be a constant scalar integer whose value is a supported kind for the intrinsic result type
print *, ieee_int(1.5, ieee_up, 3)
print *, ieee_real(1)
print *, ieee_real(1, 4)
!ERROR: 'kind=' argument must be a constant scalar integer whose value is a supported kind for the intrinsic result type
print *, ieee_real(1, 7)
end
