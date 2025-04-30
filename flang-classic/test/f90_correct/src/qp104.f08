! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for const complex convert to quad

program test
  use check_mod
  integer, parameter :: k = 16
  complex(kind = 4), parameter:: a = (1.1,1.2)
  real(kind = k) :: b,  eb
  eb = 1.10000002384185791015625000000000000_16

  b = a

  call checkr16(b, eb, 1)

end program
