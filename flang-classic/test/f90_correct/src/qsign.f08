! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test SIGN intrinsic with quad-precision arguments

program p
  use check_mod
  integer, parameter :: n = 4
  integer, parameter :: k = 16
  real(16) :: rslts(n), expect(n)
  real(kind = k) :: t1, t2

  expect(1) = 1111111111111.92222221999999999999998_16
  expect(2) = -1111111111111.92222221999999999999998_16
  expect(3) = -1.92345000000000000000000000000000007_16
  expect(4) = -123465.455999999999999999999999999997_16

  t1 = -1111111111111.92222222_16
  rslts(1) = sign(t1, 545465.7878787_16)
  t2 = -1.92345_16
  rslts(2) = sign(t1, t2)
  rslts(3) = sign(t2, t1)
  rslts(4) = sign(123465.456_16, -145678.12345_16)

  call checkr16(rslts, expect, n)

end program p
