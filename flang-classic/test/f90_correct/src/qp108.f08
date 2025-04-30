! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for constant int8 convert to quad

program main
  use check_mod
  integer, parameter :: k = 16
  real(kind = k) :: b, ea
  ea = 1.00000000000000000000000000000000000_16
  b = 1_8

  call checkr16(b, ea, 1)

end program main
