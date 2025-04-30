! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for real*4 convert to quad

program main
  use check_mod
  integer, parameter :: k = 16
  real(kind = 4) :: a = 1.1
  real(kind = k) :: b, eb
  b = a
  eb = 1.100000023841857910156250000000000_16

  call checkr16(b, eb, 1)

end program main
