! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test inverse trigonometric intrinsics (ASIN/ACOS/ATAN) with quad-precision arguments

program p
  integer, parameter :: n = 15
  integer, parameter :: k = 16
  real(16) :: rslts(n), expect(n)
  real(kind = k) :: t1
  real(kind = k) :: eps_q = 1.e-33_16
  integer(4) :: expf(15), flag(15)

  expect(1) = 0.224093090022159323812310546370480844_16
  expect(2) = -1.23450031356120897644896330052846060E-0003_16
  expect(3) = 0.00000000000000000000000000000000000_16
  expect(4) = -0.123765731093054621712607961807425668_16
  expect(5) = -0.999999999999999999999999999999999904_16
  expect(6) = 1.34670323677273729541901114526927055_16
  expect(7) = 1.57091977679521018022740957919819316_16
  expect(8) = 1.57079632679489661923132169163975140_16
  expect(9) = 1.69456205788795124094392965344717713_16
  expect(10) = 1.00000000000000000000000000000000000_16
  expect(11) = 1.57079632589489661932132169007375142_16
  expect(12) = -0.843469321333247454120837190193958463_16
  expect(13) = -0.00000000000000000000000000000000000_16
  expect(14) = -1.57078946234623197456661968049721787_16
  expect(15) = -1.00000000000000000000000000000000000_16
  expf = 1

  t1 = 0.22222222_16
  if (abs((asin(t1) - expect(1)) / asin(t1)) <= eps_q) flag(1) = 1
  t1 = -0.0012345_16
  if (abs((asin(t1) - expect(2)) / asin(t1)) <= eps_q) flag(2) = 1
  t1 = -0.0_16
  if (abs(asin(t1) - expect(3)) <= eps_q) flag(3) = 1
  if (abs((asin(-0.12345_16) - expect(4)) / asin(-0.12345_16)) <= eps_q) flag(4) = 1
  if (abs((asin(sin(-1.0_16)) - expect(5)) / asin(sin(-1.0_16))) <= eps_q) flag(5) = 1
  
  t1 = 0.22222222_16
  if (abs((acos(t1) - expect(6)) / acos(t1)) <= eps_q) flag(6) = 1 
  t1 = -0.00012345_16
  if (abs((acos(t1) - expect(7)) / acos(t1)) <= eps_q) flag(7) = 1 
  t1 = -0.0_16
  if (abs((acos(t1) - expect(8)) / acos(t1)) <= eps_q) flag(8) = 1 
  if (abs((acos(-0.12345_16) - expect(9)) / acos(-0.12345_16)) <= eps_q) flag(9) = 1 
  if (abs((acos(cos(-1.0_16)) - expect(10)) / acos(cos(-1.0_16))) <= eps_q) flag(10) = 1

  t1 = 1111111111.22222222_16
  if (abs((atan(t1) - expect(11)) / atan(t1)) <= eps_q) flag(11) = 1
  t1 = -1.12345_16
  if (abs((atan(t1) - expect(12)) / atan(t1)) <= eps_q) flag(12) = 1
  t1 = -0.0_16
  if (abs(atan(t1) - expect(13)) <= eps_q) flag(13) = 1
  if (abs((atan(-145678.12345_16) - expect(14)) / atan(-145678.12345_16)) <= eps_q) flag(14) = 1
  if (abs((atan(tan(-1.0_16)) - expect(15)) / atan(tan(-1.0_16))) <= eps_q) flag(15) = 1

  call check(flag, expf, n)

end program p
