! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test ABS intrinsic with quad-precision arguments

program p
  integer, parameter :: n = 4
  integer, parameter :: k = 16
  integer :: rslts(n), expect(n)
  real(kind = k) :: t1
  real(kind = k) :: t2 = abs(-1.123456_16), et2

  expect(1) = 1
  expect(2) = 112
  expect(3) = 11
  expect(4) = 1123
  et2 = 1.12345599999999999999999999999999997_16

  t1 = -1.341d-1
  rslts(1) = int(10 * abs(t1))
  t1 = -1.12345_16
  rslts(2) = int(100 * abs(t1))
  t1 = 1.1_16
  rslts(3) = int(10 * abs(t1))
  rslts(4) = int(1000 * abs(-1.12345_16))
  if(abs((t2 - et2) / t2) > 1e-33_16) stop 1

  call check(rslts, expect, n)

end program p
