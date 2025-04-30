! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test EPSILON/HUGE/NEAREST intrinsics with quad-precision arguments

program p
  use check_mod
  integer, parameter :: n = 7
  real(kind = 16) :: rslts(n), expect(n), t1

  expect(1) = 3.00000000000000000000000000000000039_16
  expect(2) = 2.99999999999999999999999999999999961_16
  expect(3) = 1.18973149535723176508575932662800702E4932_16
  expect(4) = 1.92592994438723585305597794258492732E-0034_16
  expect(5) = 6.47517511943802511092443895822764655E-4966_16
  expect(6) = -6.47517511943802511092443895822764655E-4966_16
  expect(7) = 0.999999999999999999999999999999999904_16

  t1 = 3.0_16
  rslts(1) = nearest(t1, 1.0_16)
  rslts(2) = nearest(t1, -1.0_16)
  rslts(3) = huge(1.0_16)
  rslts(4) = epsilon(1.0_16)
  rslts(5) = nearest(0.0_16, 1.0_16)
  rslts(6) = nearest(0.0_16, -1.0_16)
  rslts(7) = nearest(1.0_16, -1.0_16)

  call checkr16(rslts, expect, n)

end program p
