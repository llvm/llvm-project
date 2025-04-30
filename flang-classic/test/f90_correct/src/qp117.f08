! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for const quad convert to int8

program main
  use check_mod
  integer, parameter :: k = 1
  integer(kind = 8) :: b, ea
  ea = 1_8
  b = 1.1_16

  call checki8(b, ea, 1)

end program main
