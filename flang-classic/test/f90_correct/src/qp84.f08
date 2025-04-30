! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for quad convert to dcomplex

program main
  use check_mod
  integer, parameter :: k = 16
  complex(kind = 8) :: a, ea
  real(kind = k) :: b = 1.1_16
  a = b
  ea = (1.1000000000000001_8,0.0000000000000000_8)

  call checkc8(a, ea, 1)

end program main
