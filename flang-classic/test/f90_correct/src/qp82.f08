! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for quad convert to complex

program main
  use check_mod
  integer, parameter :: k = 16
  complex(kind = 4) :: a, ea
  real(kind = k) :: b = 1.1_16
  a = b
  ea = (1.10000002,0.00000000)

  call checkc4(a, ea, 1)

end program main
