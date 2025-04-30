! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test I/O for NML with quad-precision data

program test
  integer, parameter :: k = 16
  real(kind = k) :: tmpa = 1.123456789123456789123456789123456789_16
  real(kind = k) :: tmpb(3) = [1111.123456789123456789123456789123456789_16,1.1_16,1.33333333333333333333333333_16]
  character(80) :: str
  namelist /tdata/ tmpa,tmpb

  open(8,file="tnmlist.in", action='write',  delim='APOSTROPHE')

  write(unit=8,nml=tdata)
    close(8)

  tmpa = 0.0
  tmpb(1:3) = [0.0,0.0,0.0]

  open(9,file="tnmlist.in", status='OLD', recl=80, delim='APOSTROPHE')

  read(9,nml=tdata)
    close(9)

  write(str,100) tmpa
  if(str /= "   0.112345678912345678912345678912345676E+01") STOP 1
  write(str,100) tmpb(1)
  if(str /= "   0.111112345678912345678912345678912345E+04") STOP 2
  write(str,100) tmpb(2)
  if(str /= "   0.110000000000000000000000000000000008E+01") STOP 3
  write(str,100) tmpb(3)
  if(str /= "   0.133333333333333333333333332999999997E+01") STOP 4
  100 FORMAT('',e45.36)
  write(*,*) 'PASS'
end program
