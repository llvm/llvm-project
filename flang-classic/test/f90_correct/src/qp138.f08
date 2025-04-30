! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test NINT intrinsic with a quad-precision boundary value

program main
  real(16) :: a
  integer :: r, e
  a = -huge(1.0_16)
  r = nint(a)
  e = -2147483648
  call check(r, e, 1)
end
