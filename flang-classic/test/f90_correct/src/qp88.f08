! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for quad convert to logical*8

program main
  integer, parameter :: k = 16
  logical(kind = 8) :: a
  real(kind = k) :: b = 1.16_16
  character(10) :: str

  a = b
  write(str,*) a
  if(str /= "  T") STOP 1
  write(*,*) 'PASS'
end program main
