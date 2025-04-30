! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test AINT intrinsic with quad-precision arguments

program p
  use check_mod
  integer, parameter :: n = 4
  integer, parameter :: k = 16
  real(kind = k) :: rslts(n), expect(n)
  real(kind = k) :: t1
  real(kind = 4) :: a, ea

  expect(1) = 1111111111111.00000000000000000000000_16
  expect(2) = -1.00000000000000000000000000000000000_16
  expect(3) = 124.000000000000000000000000000000000_16
  expect(4) = -145678.000000000000000000000000000000_16
  ea = -1234.000_4

  t1 = 1111111111111.92222222_16
  rslts(1) = aint(t1)
  t1 = -1.92345_16
  rslts(2) = aint(t1)
  t1 = 124.1_16
  rslts(3) = aint(t1, kind = 16)
  rslts(4) = aint(-145678.12345_16)
  a = aint(-1234.784545911223344_16, kind = 4)

  if(a /= ea) STOP 1
  call checkr16(rslts, expect, n)

end program p
