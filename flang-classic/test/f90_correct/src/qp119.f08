! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for const quad convert to logical 8

program main
  logical(kind = 8) :: a
  character(10) :: str
  a = 1.1_16
  write(str,*) a
  if(str /= "  T") STOP 1
  write(*,*) 'PASS'
end program main
