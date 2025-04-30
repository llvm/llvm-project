! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test conversion to integer type with the INT intrinsic

program p
  integer, parameter :: n = 5
  integer, parameter :: k = 16
  integer(8) :: rslts(n), expect(n)
  real(kind = k) :: t1
  real(kind = k), parameter :: a = abs(2.5)
  integer(8), parameter :: c = int(a)

  expect(1) = 111111
  expect(2) = -1
  expect(3) = 124
  expect(4) = -145678
  expect(5) = 2

  t1 = 111111.92222222_16
  rslts(1) = int(t1, kind = 4)
  t1 = -1.92345_16
  rslts(2) = int(t1)
  t1 = 124.1_16
  rslts(3) = int(t1)
  rslts(4) = int(-145678.12345_16)
  rslts(5) = c

  call check(rslts, expect, n)

end program p
