! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test NINT(KIND=8) intrinsic with quad-precision arguments

program main
  use check_mod
  real(16) :: a = -huge(1.0_16), a2 = huge(1.0_16)
  integer(8) :: r(2), e(2)
  e(1) = -9223372036854775808_8
  e(2) = 9223372036854775807_8
  r(1) = nint(a, kind = 8)
  r(2) = nint(a2, kind = 8)

  call checki8(r, e, 2)
end
