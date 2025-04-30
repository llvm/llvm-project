! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test quad-precision division

program main
  use check_mod
  real(16) :: a = 6.5421_16/1.23456_16, e
  e = 5.29913491446345256609642301710730960_16
  call checkr16(a, e, 1)
end
