!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program test
  use ieee_arithmetic
  use check_mod
  real (16) , dimension(16) :: c , ec
  real (16) :: c1, c2, c3
  c3 = 0._16
  c1 = 1.0 / c3
  c2 = -1.0 / c3
  c(1) = 2.0_16
  c(2) = 2.0_16
  c(3) = 2.0_16
  c(4) = 1.0_16 / c3
  c(5) = -1.0_16 / c3
  c(6) = 0.0_16
  c(7) = 0.0_16
  c(8) = 1.99999999999999999999999999999999981_16
  c(9) = 2.00000000000000000000000000000000039_16
  c(10) = -tiny(1.0_16)
  c(11) = -tiny(1.0_16)
  c(12) = -huge(1.0_16)
  c(15) = sqrt(-1.0_16)
  c(16) = sqrt(-1.0_16)
  ec(1) = 2.00000000000000000000000000000000000_16
  ec(2) = 2.00000000000000000000000000000000039_16
  ec(3) = 1.99999999999999999999999999999999981_16
  ec(4) = 1.18973149535723176508575932662800702E+4932_16
  ec(5) = -1.18973149535723176508575932662800702E+4932_16
  ec(6) = 6.47517511943802511092443895822764655E-4966_16
  ec(7) = -6.47517511943802511092443895822764655E-4966_16
  ec(8) = 2.00000000000000000000000000000000000_16
  ec(9) = 2.00000000000000000000000000000000000_16
  ec(10) = -3.36210314311209350626267781732175196E-4932_16
  ec(11) = -3.36210314311209350626267781732175325E-4932_16
  ec(12) = -1.18973149535723176508575932662800690E+4932_16
  ec(13) = 1.189731495357231765085759326628007E+4932_16
  ec(14) = -1.18973149535723176508575932662800702E+4932_16

  !these are test for normal 
  c(1) = ieee_next_after(c(1), 2.0_16)
  c(2) = ieee_next_after(c(2), 3.0_16)
  c(3) = ieee_next_after(c(3), 1.0_16)
  !these are test for inf , -inf
  c(4) = ieee_next_after(c1, c2)
  c(5) = ieee_next_after(c2, c1)
  !these are test for 0
  c(6) = ieee_next_after(c(6), 1.0_16)
  c(7) = ieee_next_after(c(7), -1.0_16)
  !these are test for mantissa is non-zero
  c(8) = ieee_next_after(c(8), 3.0_16)
  c(9) = ieee_next_after(c(9), 1.0_16)
  !these are test for underflow
  c(10) = ieee_next_after(c(10), 1.0_16)
  c(11) = ieee_next_after(c(11), -1.0_16)
  !this is test for overflow
  c(12) = ieee_next_after(c(12), 1.0_16)
  !these are test for inf , const 
  c(13) = ieee_next_after(c1, 1.0_16)
  c(14) = ieee_next_after(c2, 2.0_16)
  !these are test for nan
  c(15) = ieee_next_after(c(15), 1.0_16)
  c(16) = ieee_next_after(1.0_16, c(16))

  !call checkr16(c(3), ec(3), 1, rtoler = 1e-36_16)
  !stop 1
  if (ieee_is_finite(ieee_next_after(c1, c1))) STOP 1
  if (ieee_is_finite(ieee_next_after(c2, c2))) STOP 2
  if (ieee_is_finite(ieee_next_after(huge(1.0_16), c1))) STOP 3
  if (ieee_is_nan(c(15)) .neqv. .true.) STOP 4
  if (ieee_is_nan(c(16)) .neqv. .true.) STOP 5
  call checkr16(c, ec, 14)
 
end program test
