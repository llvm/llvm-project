! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test QMIN intrinsic with quad-precision arguments

program main
  use check_mod
  real(16) :: a, e
  a = qmin(1.23456789_16, 2.4567981321654_16, &
          -1.45645465_16, -huge(1.0_16), -tiny(1.0_16))
  e = -huge(1.0_16)
  call checkr16(a, e, 1)
end
