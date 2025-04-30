! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test conversion of real*16 to real*4

program main
  use check_mod
  integer, parameter :: k = 16
  real(kind = k) :: a = 1.19_16
  real(kind = 4) :: b, eb
  b = a
  eb = 1.19000006

  call checkr4(b, eb, 1)

end program main
