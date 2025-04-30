! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test exponentiation of quad-precision values

program test
  use check_mod
  integer, parameter :: k = 16
  real(kind = k) :: a = 1.1_16 ** 1.2_16, ea

  ea = 1.12116936414060228271727326177460416_16

  call checkr16(a, ea, 1)
end program
