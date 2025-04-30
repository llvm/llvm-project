! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test SIND intrinsic with quad-precision arguments

program test
  real(16) :: r1, r2, q1 = -1.23456789_16
  real(16) :: eps_q = 1.e-33_16
  integer :: result(2), expect(2)

  expect = 1
  result = 0

  r1 = sind(9.87654321_16)
  r2 = sind(q1)

  if (abs((r1 - 0.171525783880303950850296264507163812_16)/r1) <= eps_q) result(1) = 1
  if (abs((r2 - (-2.15456072124056281482229821966887541E-0002_16))/r2) <= eps_q) result(2) = 1

  call check(result, expect, 2)

end program test
