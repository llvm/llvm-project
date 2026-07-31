! REQUIRES: flang-supports-f128-math
! RUN: %python %S/test_folding.py %s %flang_fc1

! z**(1,0) must fold to z exactly at quad precision.

real(16), parameter :: &
  r_cplx16 = real((3.0_16, 0.0_16) ** (1.0_16, 0.0_16), kind=16)
logical, parameter :: test_cpow_unity_cplx16 = r_cplx16 == 3.0_16

complex(16), parameter :: z16 = (3.0_16, 0.0_16)
real(16), parameter :: r_var16 = real(z16 ** (1.0_16, 0.0_16), kind=16)
logical, parameter :: test_cpow_unity_var16 = r_var16 == 3.0_16

end
