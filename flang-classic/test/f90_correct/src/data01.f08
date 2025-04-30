! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! check data statement for real*16 variable

program test
  use check_mod
  real(16) x, y
  data x / -1.e0 /

  y = -1.0_16
  call checkr16(x, y, 1)

end program
