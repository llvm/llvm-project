! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for quad convert to int8

program main
  use check_mod
  integer, parameter :: k = 16
  integer(kind = 8) :: a, ea
  real(kind = k) :: b = 2.1_16
  a = b
  ea = 2_8

  call checki8(a, ea, 1)

end program main
