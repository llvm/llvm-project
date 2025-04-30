! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test REAL/FLOAT/FLOOR/CEILING/SNGL/AINT/ANINT intrinsics returning double-precision results

program main
  use check_mod
  real*4 a
  real*8 rsltb(15), expctb(15)
  real*16 c
  integer*2 d
  integer*4 e
  integer*8 f

  PARAMETER(a = 3.1415927_4)
  PARAMETER(c = 3.1415927_16)
  PARAMETER(d = 123_2)
  PARAMETER(e = 123_4)
  PARAMETER(f = 123_8)

  expctb(1) = 3.141592741012573_8
  expctb(2) = 3.141592741012573_8
  expctb(3) = 123.0000000000000_8
  expctb(4) = 123.0000000000000_8
  expctb(5) = 123.0000000000000_8
  expctb(6) = 123.0000000000000_8
  expctb(7) = 3.000000000000000_8
  expctb(8) = 3.000000000000000_8
  expctb(9) = 4.000000000000000_8
  expctb(10) = 4.000000000000000_8
  expctb(11) = 3.141592741012573_8
  expctb(12) = 3.000000000000000_8
  expctb(13) = 3.000000000000000_8
  expctb(14) = 3.000000000000000_8
  expctb(15) = 3.000000000000000_8

  rsltb(1) = real(a)
  rsltb(2) = real(c)
  rsltb(3) = real(d)
  rsltb(4) = real(e)
  rsltb(5) = real(f)
  rsltb(6) = float(e)
  rsltb(7) = floor(a)
  rsltb(8) = floor(c)
  rsltb(9) = ceiling(a)
  rsltb(10) = ceiling(c)
  rsltb(11) = sngl(c)
  rsltb(12) = aint(a)
  rsltb(13) = aint(c)
  rsltb(14) = anint(a)
  rsltb(15) = anint(c)

  call checkr8(rsltb, expctb, 15, rtoler = 5.d-15)

end program
