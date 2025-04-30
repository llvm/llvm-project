! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for const quad convert to blog

program main
  use check_mod
  integer, parameter :: k = 1
  logical(kind = 1) :: b, ea
  ea = .TRUE.
  b = 1.1_16

  call checkl1(b, ea, 1)

end program main
