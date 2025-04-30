! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test FLOOR intrinsic with quad-precision arguments

program p
  integer, parameter :: n = 8
  integer, parameter :: k = 16
  integer(8) :: rslts(n), expect(n)
  integer(8), parameter :: r1 = floor(1.23456_16 + sin(0.0_16))
  integer(8), parameter :: r2 = floor(-1.23456_16 + sin(0.0_16))
  integer(8) :: r3(2) = floor((/1.23456_16, -1.23456_16/))
  real(kind = k) :: t1

  expect(1) = 2147483647
  expect(2) = -2
  expect(3) = 124
  expect(4) = -145679
  expect(5) = 1
  expect(6) = -2
  expect(7) = 1
  expect(8) = -2

  t1 = 1111111111111.22222222
  rslts(1) = floor(t1)
  t1 = -1.12345_16
  rslts(2) = floor(t1)
  t1 = 124.1_16
  rslts(3) = floor(t1)
  rslts(4) = floor(-145678.12345_16)
  rslts(5) = r1
  rslts(6) = r2
  rslts(7) = r3(1)
  rslts(8) = r3(2)

  call check(rslts, expect, n)

end program p
