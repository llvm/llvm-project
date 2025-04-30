! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test INT/NINT/IFIX/IDINT/IDNINT intrinsics returning 32-bit results

program main
  use check_mod
  real*4 a
  real*8 b
  real*16 c
  integer*2 d
  integer*4 rslte(11), expcte(11)
  integer*8 f

  PARAMETER(a = 3.1415927_4)
  PARAMETER(b = 3.1415927_8)
  PARAMETER(c = 3.1415927_16)
  PARAMETER(d = 123_2)
  PARAMETER(f = 123_8)

  expcte(1) = 3_4
  expcte(2) = 3_4
  expcte(3) = 3_4
  expcte(4) = 123_4
  expcte(5) = 123_4
  expcte(6) = 3_4
  expcte(7) = 3_4
  expcte(8) = 3_4
  expcte(9) = 3_4
  expcte(10) = 3_4
  expcte(11) = 3_4

  rslte(1) = int(a, kind = 4)
  rslte(2) = int(b, kind = 4)
  rslte(3) = int(c, kind = 4)
  rslte(4) = int(d, kind = 4)
  rslte(5) = int(f, kind = 4)
  rslte(6) = nint(a, kind = 4)
  rslte(7) = nint(b, kind = 4)
  rslte(8) = nint(c, kind = 4)
  rslte(9) = ifix(a)
  rslte(10) = idint(b)
  rslte(11) = idnint(b)

  call checki4(rslte, expcte, 11)

end program
