! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for const char convert to quad

program test
  use check_mod
  integer, parameter :: k = 16
  real(kind = k) ::  b, ea
  b = 'a'
  ea = 8.86353765139443853322583968516834914E-2457_16

  call checkr16(b, ea, 1)

end program test
