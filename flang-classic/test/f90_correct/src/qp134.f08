! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! check that TAND(90.0) returns infinity

program main
  use ieee_arithmetic
  real(8) :: a
  a = tand(90.0_8)
  if(ieee_is_finite(a)) STOP 1
  write(*,*) 'PASS'
end program main
