! RUN: %python %S/test_folding.py %s %flang_fc1

! z**(1,0) must fold to z exactly.

real(8), parameter :: &
  r_cplx = real((3.0_8, 0.0_8) ** (1.0_8, 0.0_8), kind=8)
logical, parameter :: test_cpow_unity_cplx = r_cplx == 3.0_8

real(8), parameter :: &
  r_real = real(3.0_8 ** (1.0_8, 0.0_8), kind=8)
logical, parameter :: test_cpow_unity_real = r_real == 3.0_8

complex(8), parameter :: z = (3.0_8, 0.0_8)
real(8), parameter :: r_var = real(z ** (1.0_8, 0.0_8), kind=8)
logical, parameter :: test_cpow_unity_var = r_var == 3.0_8

end
