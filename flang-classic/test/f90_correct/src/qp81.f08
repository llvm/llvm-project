! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for quad convert to blogical

program main
  use check_mod
  integer, parameter :: k = 16
  logical(kind = 1) :: a, ea
  real(kind = k) :: b = 1.16_16
  a = b
  ea = .TRUE.

  call checkl1(a, ea, 1)

end program main
