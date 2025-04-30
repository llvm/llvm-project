! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test RRSPACING intrinsic with quad-precision arguments

program p
  use ieee_arithmetic
  use check_mod
  integer, parameter :: n = 2
  integer, parameter :: k = 16
  real(kind = k) :: z = 0.0_16, pz = -1.0_16
  real(kind = k) :: rslts(n), expect(n)
  real(kind = k) :: t1, a16(2), ea16(2)

  expect(1) = 6346140593337462971459878561201237.00_16
  expect(2) = 5770905019734831278942385760024238.00_16
  a16 = [1.0_16/z, sqrt(pz)]

  t1 = 1.22222222_16
  rslts(1) = rrspacing(t1)
  rslts(2) = rrspacing(-145678.12345_16)
  a16 = rrspacing(a16)

  if(ieee_is_nan(a16(1)) .eq. .false.) STOP 1
  if(ieee_is_nan(a16(2)) .eq. .false.) STOP 2
  call checkr16(rslts, expect, 2)
end program p
