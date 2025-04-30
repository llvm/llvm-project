! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test trigonometric intrinsics (SIN/COS/TAN) with quad-precision arguments

program p
  integer, parameter :: n = 15
  integer, parameter :: k = 16
  real(16) :: rslts(n), expect(n)
  real(kind = k) :: t1
  real(kind = k) :: eps_q = 1.e-33_16
  integer(4) :: expf(15), flag(15)

  expect(1) = -0.891989585462173237264341997484966556_16
  expect(2) = -0.901598186916934690816538731485821062_16
  expect(3) = -0.00000000000000000000000000000000000_16
  expect(4) = -0.620585853236383773851092144132331337_16
  expect(5) = -1.00000000000000000000000000000000000_16
  expect(6) = -0.452055947231114687140625088242801529_16
  expect(7) = 0.432574513058844112152353951825942533_16
  expect(8) = 1.00000000000000000000000000000000000_16
  expect(9) = -0.784138507384294665135583052875327117_16
  expect(10) = -1.00000000000000000000000000000000000_16
  expect(11) = 1.97318405150002688169710600636888432_16
  expect(12) = -2.08426099943222545525046113959937295_16
  expect(13) = -0.00000000000000000000000000000000000_16
  expect(14) = 0.791423769388032159749269167724384173_16
  expect(15) = -1.00000000000000000000000000000000000_16
  expf = 1

  t1 = 1111111111.22222222_16
  if (abs((sin(t1) - expect(1)) / sin(t1)) <= eps_q) flag(1) = 1
  t1 = -1.12345_16
  if (abs((sin(t1) - expect(2)) / sin(t1)) <= eps_q) flag(2) = 1
  t1 = -0.0_16
  if (abs(sin(t1) - expect(3)) <= eps_q) flag(3) = 1
  if (abs((sin(-145678.12345_16) - expect(4)) / sin(-145678.12345_16)) <= eps_q) flag(4) = 1
  if (abs((sin(asin(-1.0_16)) - expect(5)) / sin(asin(-1.0_16))) <= eps_q) flag(5) = 1

  t1 = 1111111111.22222222_16
  if (abs((cos(t1) - expect(6)) / cos(t1)) <= eps_q) flag(6) = 1
  t1 = -1.12345_16
  if (abs((cos(t1) - expect(7)) / cos(t1)) <= eps_q) flag(7) = 1
  t1 = -0.0_16
  if (abs((cos(t1) - expect(8)) / cos(t1)) <= eps_q) flag(8) = 1
  if (abs((cos(-145678.12345_16) - expect(9)) / cos(-145678.12345_16)) <= eps_q) flag(9) = 1
  if (abs((cos(acos(-1.0_16)) - expect(10)) / cos(acos(-1.0_16))) <= eps_q) flag(10) = 1

  t1 = 1111111111.22222222_16
  if (abs((tan(t1) - expect(11)) / tan(t1)) <= eps_q) flag(11) = 1
  t1 = -1.12345_16
  if (abs((tan(t1) - expect(12)) / tan(t1)) <= eps_q) flag(12) = 1
  t1 = -0.0_16
  if (abs(tan(t1) - expect(13)) <= eps_q) flag(13) = 1
  if (abs((tan(-145678.12345_16) - expect(14)) / tan(-145678.12345_16)) <= eps_q) flag(14) = 1
  if (abs((tan(atan(-1.0_16)) - expect(15)) / tan(atan(-1.0_16))) <= eps_q) flag(15) = 1

  call check(flag, expf, n)

end program p
