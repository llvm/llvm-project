! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test SHAPE intrinsic with quad-precision arguments

program main
  use check_mod
  real(16), parameter :: a(30,30,30) = 1.0_16
  integer :: b(3) = shape(a(3:27:2, 1:29, 30:4:-6))
  integer, parameter :: c(3) = shape(a(3:27:2, 1:29, 30:4:-6))
  integer :: d(3), e(0), ee(1), expct(3)
  integer(8) :: f(3), expct8(3)
  expct = [13, 29, 5]
  expct8 = [13_8, 29_8, 9_8]
  ee = 1
  d = shape(a(3:27:2, 1:29, 30:4:-6))
  e = shape(huge(1.0_16))
  f = shape(a(3:27:2, 1:29, 30:4:-3), kind = 8)

  if(dot_product(ee, e) /= 0) stop 1
  if(any(b /= expct)) stop 2
  if(any(c /= expct)) stop 3
  if(any(d /= expct)) stop 4

  call checki8(f, expct8, 3)
end
