! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test exponentiation of quad-precision value with a negative INTEGER*8 exponent

program main
  use check_mod
  integer(8) :: b = -4_8
  real(16) :: a = 2.0_16, ea
  a = a ** (b)
  ea = 6.25000000000000000000000000000000000E-0002_16

  call checkr16(a, ea, 1)
end
