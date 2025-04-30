! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test I/O for NML

program test
  integer, parameter :: k = 16
  real(kind = k) :: qtmp = 1.11_16
  real(kind = 8) :: dtmp = 1.12_16
  real(kind = 4) :: rtmp = 1.13_16
  integer(kind = 8) :: ltmp = 1.13e11_16
  integer(kind = 4) :: tmp = 65536.1_16
  integer(kind = 2) :: stmp = 257.1_16
  integer(kind = 1) :: btmp = 1.13_16
  complex(kind=4) :: rctmp = (1.11_16, 1.12_16)
  complex(kind=8) :: dctmp = (1.13_16, 1.14_16)
  complex(kind=16) :: qctmp = (1.15_16, 1.16_16)
  character(80) :: str
  namelist /tdata/ qtmp,dtmp,rtmp,ltmp,tmp,stmp,btmp,rctmp,dctmp,qctmp

  open(8, file="tnmlist.in", action='write',  delim='APOSTROPHE')

  write(unit=8, nml=tdata)
  close(8)

  qtmp = 0.0
  dtmp = 0.0
  rtmp = 0.0
  ltmp = 0
  tmp = 0
  stmp = 0
  btmp = 0
  rctmp = (0.0,0.0)
  qctmp = (0.0,0.0)

  open(9, file="tnmlist.in", status='OLD', recl=80, delim='APOSTROPHE')

  read(9,nml=tdata)
  close(9)

  write(str,*) qtmp
  if(str /= "   1.11000000000000000000000000000000008") STOP 1

  write(str,*) dtmp
  if(str /= "    1.120000000000000") STOP 2

  write(str,*) rtmp
  if(str /= "    1.130000") STOP 3

  write(str,*) ltmp
  if(str /= "             113000000000") STOP 4

  write(str,*) tmp
  if(str /= "        65536") STOP 5

  write(str,*) stmp
  if(str /= "     257") STOP 6

  write(str,*) btmp
  if(str /= "     1") STOP 7

  write(str,*) rctmp
  if(str /= " (1.110000,1.120000)") STOP 8

  write(str,*) dctmp
  if(str /= " (1.130000000000000,1.140000000000000)") STOP 9

  write(str,*) qctmp
  if(str /= " (1.14999999999999999999999999999999992,1.15999999999999999999999999999999993)") STOP 10

  write(*,*) 'PASS'
end program
