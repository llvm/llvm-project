! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for quad convert to double

program main
  use check_mod
  integer, parameter :: k = 16
  real(kind = k) :: a = 1.19_16
  real(kind = 8) :: b, eb
  b = a
  eb = 1.1899999999999999_8

  call checkr8(b, eb, 1)

end program main
