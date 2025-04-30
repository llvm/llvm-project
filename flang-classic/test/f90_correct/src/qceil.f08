! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test CEILING intrinsic with quad-precision arguments

program p
  integer, parameter :: n = 8
  integer, parameter :: k = 16
  integer(8) :: rslts(n), expect(n)
  integer(8), parameter :: r1 = ceiling(1.23456_16 + sin(0.0_16))
  integer(8), parameter :: r2 = ceiling(-1.23456_16 + sin(0.0_16))
  integer(8) :: r3(2) = ceiling((/1.23456_16, -1.23456_16/))
  real(kind = k) :: t1

  expect(1) = 111112
  expect(2) = -1
  expect(3) = 125
  expect(4) = -145678
  expect(5) = 2
  expect(6) = -1
  expect(7) = 2
  expect(8) = -1

  t1 = 111111.22222222_16
  rslts(1) = ceiling(t1)
  t1 = -1.12345_16
  rslts(2) = ceiling(t1)
  t1 = 124.9_16
  rslts(3) = ceiling(t1)
  rslts(4) = ceiling(-145678.12345_16)
  rslts(5) = r1
  rslts(6) = r2
  rslts(7) = r3(1)
  rslts(8) = r3(2)

  call check(rslts, expect, n)

end program p
