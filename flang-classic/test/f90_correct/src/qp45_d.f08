! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test INT/NINT/IFIX/IDINT/IDNINT/INT2 intrinsics returning 16-bit results

program main
  use check_mod
  real*4 a
  real*8 b
  real*16 c
  integer*2 rsltd(16), expctd(16)
  integer*4 e
  integer*8 f

  PARAMETER(a = 3.1415927_4)
  PARAMETER(b = 3.1415927_8)
  PARAMETER(c = 3.1415927_16)
  PARAMETER(e = 123_4)
  PARAMETER(f = 123_8)

  expctd(1) = 3_2
  expctd(2) = 3_2
  expctd(3) = 3_2
  expctd(4) = 123_2
  expctd(5) = 123_2
  expctd(6) = 3_2
  expctd(7) = 3_2
  expctd(8) = 3_2
  expctd(9) = 3_2
  expctd(10) = 3_2
  expctd(11) = 3_2
  expctd(12) = 3_2
  expctd(13) = 3_2
  expctd(14) = 3_2
  expctd(15) = 123_2
  expctd(16) = 123_2

  rsltd(1) = int(a, kind = 2)
  rsltd(2) = int(b, kind = 2)
  rsltd(3) = int(c, kind = 2)
  rsltd(4) = int(e, kind = 2)
  rsltd(5) = int(f, kind = 2)
  rsltd(6) = nint(a, kind = 2)
  rsltd(7) = nint(b, kind = 2)
  rsltd(8) = nint(c, kind = 2)
  rsltd(9) = ifix(a)
  rsltd(10) = idint(b)
  rsltd(11) = idnint(b)
  rsltd(12) = int2(a)
  rsltd(13) = int2(b)
  rsltd(14) = int2(c)
  rsltd(15) = int2(e)
  rsltd(16) = int2(f)

  call checki2(rsltd, expctd, 16)

end program
