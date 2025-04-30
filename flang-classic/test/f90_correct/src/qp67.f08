! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for const quad convert to real *4

program main
  use check_mod
  real(kind = 4) :: a = 1.1_16, ea
  ea = 1.10000002

  call checkr4(a, ea, 1)

end program main
