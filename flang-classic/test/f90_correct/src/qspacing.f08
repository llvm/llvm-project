! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test SPACING intrinsic with quad-precision arguments

program p
  use check_mod
  integer, parameter :: n = 2
  integer, parameter :: k = 16
  real(kind = k) :: rslts(n), expect(n)
  real(kind = k) :: t1

  expect(1) = 1.92592994438723585305597794258492732E-0034_16
  expect(2) = 2.52435489670723777731753140890491593E-0029_16

  t1 = 1.22222222_16
  rslts(1) = spacing(t1)
  rslts(2) = spacing(-145678.12345_16)

  call checkr16(rslts, expect, n)

end program p
