! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for const double convert to quad

program main
  use check_mod
  integer, parameter :: k = 16
  real(kind = k) ::  b, ea
  ea = 1.10000000000000008881784197001252323_16
  b = 1.1_8

  call checkr16(b, ea, 1)

end program main
