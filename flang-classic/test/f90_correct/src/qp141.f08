! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test NINT(KIND=8) intrinsic with a quad-precision argument

program main
  use check_mod
  integer(8) :: r = nint(123456123.789_16, kind = 8), e
  e = 123456124_8
  call checki8(r, e, 1)
end
