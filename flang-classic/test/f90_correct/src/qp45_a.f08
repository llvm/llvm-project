! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test REAL/FLOAT/FLOOR/CEILING/SNGL/AINT/ANINT intrinsics returning single-precision results

program main
  use check_mod
  real*4 rslta(15), expcta(15)
  real*8 b
  real*16 c
  integer*2 d
  integer*4 e
  integer*8 f

  PARAMETER(b = 3.1415927_8)
  PARAMETER(c = 3.1415927_16)
  PARAMETER(d = 123_2)
  PARAMETER(e = 123_4)
  PARAMETER(f = 123_8)

  expcta(1) = 3.141593_4
  expcta(2) = 3.141593_4
  expcta(3) = 123.0000_4
  expcta(4) = 123.0000_4
  expcta(5) = 123.0000_4
  expcta(6) = 123.0000_4
  expcta(7) = 3.000000_4
  expcta(8) = 3.000000_4
  expcta(9) = 4.000000_4
  expcta(10) = 4.000000_4
  expcta(11) = 3.141593_4
  expcta(12) = 3.000000_4
  expcta(13) = 3.000000_4
  expcta(14) = 3.000000_4
  expcta(15) = 3.000000_4

  rslta(1) = real(b)
  rslta(2) = real(c)
  rslta(3) = real(d)
  rslta(4) = real(e)
  rslta(5) = real(f)
  rslta(6) = float(e)
  rslta(7) = floor(b)
  rslta(8) = floor(c)
  rslta(9) = ceiling(b)
  rslta(10) = ceiling(c)
  rslta(11) = sngl(c)
  rslta(12) = aint(b)
  rslta(13) = aint(c)
  rslta(14) = anint(b)
  rslta(15) = anint(c)

  call checkr4(rslta, expcta, 15, rtoler = 5.e-6)

end program
