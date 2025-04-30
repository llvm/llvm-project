! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test quad-precision addition

program main
  use check_mod
  real(16) :: a = 1.23456_16+6.5421_16, e
  e = 7.77666000000000000000000000000000005_16
  call checkr16(a, e, 1)
end
