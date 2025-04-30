! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test NINT intrinsic with quad-precision arguments

program p
  integer, parameter :: n = 8
  integer, parameter :: k = 16
  integer(4) :: rslts(n), expect(n)
  real(kind = k) :: t1
  integer(4), parameter :: c = nint(1.2345_16 + sin(0.0_16))
  integer(4), parameter :: d = nint(huge(0.0_16) + sin(0.0_16))
  integer(4), parameter :: e = nint(-1.2345_16 + sin(0.0_16))
  integer(4), parameter :: f = nint(-huge(0.0_16) + sin(0.0_16))

  expect(1) = 2147483647
  expect(2) = -2
  expect(3) = 124
  expect(4) = -145678
  expect(5) = 1
  expect(6) = 2147483647
  expect(7) = -1
  expect(8) = -2147483648

  t1 = 1111111111111.92222222_16
  rslts(1) = nint(t1)
  t1 = -1.92345_16
  rslts(2) = nint(t1)
  t1 = 124.1_16
  rslts(3) = nint(t1, kind = 8)
  rslts(4) = nint(-145678.12345_16)
  rslts(5) = c
  rslts(6) = d
  rslts(7) = e
  rslts(8) = f

  call check(rslts, expect, n)

end program p
