! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test conversion of quad real to double complex

program main
  use check_mod
  complex(kind = 8) :: b, ea
  ea = (1.1000000000000001_8, 0.0000000000000000_8)
  b = 1.1_16

  call checkc8(b, ea, 1)

end program main
