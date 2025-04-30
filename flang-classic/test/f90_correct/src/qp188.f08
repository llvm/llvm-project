! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test I/O of quad-precision values

program test
  use ieee_arithmetic
  integer, parameter :: qp = 8, n = 22
  real(kind = qp) :: r(n)
  character(80) :: s(n)
  integer :: result(n), expect(n), i
  expect = 1
  result = 0
  r(1) = 123456789.123456789_qp
  r(2) = 123456789123456789._qp
  r(3) = 0.123456789123456789_qp
  r(4) = 0.00000000000000001234567891234567899_qp
  r(5) = 1.23456789123456789_qp
  r(6) = 1.23456789123456789e+9_qp
  r(7) = -1.2345678912345678e+99_qp
  r(8) = 1.23456789123456789e+123_qp
  r(9) = 1.23456789123456789e-3_qp
  r(10) = -1.23456789123456789e-33_qp
  r(11) = 1.23456789123456789e-111_qp
  r(12) = huge(0.0_qp)
  r(13) = -tiny(0.0_qp)
  r(14) = epsilon(0.0_qp)
  r(15) = ieee_value(r(15), ieee_positive_zero)
  r(16) = ieee_value(r(16), ieee_negative_zero)
  r(17) = ieee_value(r(17), ieee_positive_inf)
  r(18) = ieee_value(r(18), ieee_negative_inf)
  r(19) = ieee_value(r(19), ieee_signaling_nan)
  r(20) = ieee_value(r(20), ieee_quiet_nan)
  r(21) = ieee_value(r(21), ieee_positive_denormal)
  r(22) = ieee_value(r(22), ieee_negative_denormal)
  
  do i = 1, n
    write(s(i), *) r(i)
  enddo
 
  if(s(1) == "    123456789.1234568") result(1) = 1
  if(s(2) == "   1.2345678912345678E+017") result(2) = 1
  if(s(3) == "   0.1234567891234568") result(3) = 1
  if(s(4) == "   1.2345678912345679E-017") result(4) = 1
  if(s(5) == "    1.234567891234568") result(5) = 1
  if(s(6) == "    1234567891.234568") result(6) = 1
  if(s(7) == "  -1.2345678912345679E+099") result(7) = 1
  if(s(8) == "   1.2345678912345679E+123") result(8) = 1
  if(s(9) == "   1.2345678912345679E-003") result(9) = 1
  if(s(10) == "  -1.2345678912345679E-033") result(10) = 1
  if(s(11) == "   1.2345678912345678E-111") result(11) = 1
  if(s(12) == "   1.7976931348623157E+308") result(12) = 1
  if(s(13) == "  -2.2250738585072014E-308") result(13) = 1
  if(s(14) == "   2.2204460492503131E-016") result(14) = 1
  if(s(15) == "    0.000000000000000") result(15) = 1
  if(s(16) == "   -0.000000000000000") result(16) = 1
  if(s(17) == "                       Inf") result(17) = 1
  if(s(18) == "                      -Inf") result(18) = 1
  if(s(19) == "                       NaN") result(19) = 1
  if(s(20) == "                       NaN") result(20) = 1
  if(s(21) == "   1.1125369292536007E-308") result(21) = 1
  if(s(22) == "  -1.1125369292536007E-308") result(22) = 1
  call check(result, expect, n)
end program
