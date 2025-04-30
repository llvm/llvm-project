! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test REAL/FLOAT/FLOOR/CEILING/SNGL/AINT/ANINT intrinsics returning quad-precision results

program main
  use check_mod
  real*4 a
  real*8 b
  real*16 rsltc(15), expctc(15)
  integer*2 d
  integer*4 e
  integer*8 f

  PARAMETER(a = 3.1415927_4)
  PARAMETER(b = 3.1415927_8)
  PARAMETER(d = 123_2)
  PARAMETER(e = 123_4)
  PARAMETER(f = 123_8)

  expctc(1) = 3.14159274101257324218750000000000000_16
  expctc(2) = 3.14159274101257324218750000000000000_16
  expctc(3) = 123.000000000000000000000000000000000_16
  expctc(4) = 123.000000000000000000000000000000000_16
  expctc(5) = 123.000000000000000000000000000000000_16
  expctc(6) = 123.000000000000000000000000000000000_16
  expctc(7) = 3.00000000000000000000000000000000000_16
  expctc(8) = 3.00000000000000000000000000000000000_16
  expctc(9) = 4.00000000000000000000000000000000000_16
  expctc(10) = 4.00000000000000000000000000000000000_16
  expctc(11) = 3.14159274101257324218750000000000000_16
  expctc(12) = 3.00000000000000000000000000000000000_16
  expctc(13) = 3.00000000000000000000000000000000000_16
  expctc(14) = 3.00000000000000000000000000000000000_16
  expctc(15) = 3.00000000000000000000000000000000000_16

  rsltc(1) = real(a)
  rsltc(2) = real(b)
  rsltc(3) = real(d)
  rsltc(4) = real(e)
  rsltc(5) = real(f)
  rsltc(6) = float(e)
  rsltc(7) = floor(a)
  rsltc(8) = floor(b)
  rsltc(9) = ceiling(a)
  rsltc(10) = ceiling(b)
  rsltc(11) = sngl(b)
  rsltc(12) = aint(a)
  rsltc(13) = aint(b)
  rsltc(14) = anint(a)
  rsltc(15) = anint(b)

  call checkr16(rsltc, expctc, 15)

end program
