! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test INT/NINT/IFIX/IDINT/IDNINT/INT8 intrinsics returning 64-bit results

program main
  use check_mod
  real*4 a
  real*8 b
  real*16 c
  integer*2 d
  integer*4 e
  integer*8 rsltf(16), expctf(16)

  PARAMETER(a = 3.1415927_4)
  PARAMETER(b = 3.1415927_8)
  PARAMETER(c = 3.1415927_16)
  PARAMETER(d = 123_2)
  PARAMETER(e = 123_4)

  expctf(1) = 3_8
  expctf(2) = 3_8
  expctf(3) = 3_8
  expctf(4) = 123_8
  expctf(5) = 123_8
  expctf(6) = 3_8
  expctf(7) = 3_8
  expctf(8) = 3_8
  expctf(9) = 3_8
  expctf(10) = 3_8
  expctf(11) =  3_8
  expctf(12) = 3_8
  expctf(13) = 3_8
  expctf(14) = 3_8
  expctf(15) = 123_8
  expctf(16) = 123_8

  rsltf(1) = int(a, kind = 8)
  rsltf(2) = int(b, kind = 8)
  rsltf(3) = int(c, kind = 8)
  rsltf(4) = int(e, kind = 8)
  rsltf(5) = int(d, kind = 8)
  rsltf(6) = nint(a, kind = 8)
  rsltf(7) = nint(b, kind = 8)
  rsltf(8) = nint(c, kind = 8)
  rsltf(9) = ifix(a)
  rsltf(10) = idint(b)
  rsltf(11) = idnint(b)
  rsltf(12) = int8(a)
  rsltf(13) = int8(b)
  rsltf(14) = int8(c)
  rsltf(15) = int8(e)
  rsltf(16) = int8(d)

  call checki8(rsltf, expctf, 16)

end program
