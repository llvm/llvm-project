! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for const logical8 convert to quad

program main
  use check_mod
  integer, parameter :: k = 16
  logical(kind = 8), parameter :: a = .TRUE.
  real(kind = k) :: b = a, eb
  eb = -1.00000000000000000000000000000000000_16

  call checkr16(b, eb, 1)

end program main
