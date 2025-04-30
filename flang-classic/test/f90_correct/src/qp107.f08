! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for constant hollerith convert to quad

program main
  use check_mod
  integer, parameter :: k = 16
  real(kind = k):: c, eb
  eb = 8.86353765139443853322584019599117077E-2457_16
  c = 4h1234

  call checkr16(c, eb, 1)

end program main
