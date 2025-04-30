! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test formatted file I/O of quad-precision values

program test
  integer, parameter :: k = 16
  real(kind = k) :: a, e = 1.23456_16
  character(50) :: str

  open(8,file = "r.in", action = 'write',  delim = 'APOSTROPHE')
  write(unit = 8,FMT = 100) e
  close(8)
  open(9,file = "r.in", status = 'OLD', recl = 80, delim = 'APOSTROPHE')
  read(unit = 9,FMT = 100) a
  close(9)
  write(str,*) a
  100 FORMAT(F45.35)
  if(str /= "   1.23456000000000000000000000000000005") STOP 1
  write(*,*) 'PASS'

end program
