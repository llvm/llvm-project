! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test I/O for NML

program test
  use ieee_arithmetic
  integer, parameter :: k = 16
  real(kind = k) :: qtmp
  real(kind = 8) :: dtmp
  real(kind = 4) :: rtmp, inf, ninf, nan
  integer(kind = 8) :: ltmp
  integer(kind = 4) :: tmp
  integer(kind = 2) :: stmp
  integer(kind = 1) :: btmp
  complex(kind=4) :: rctmp
  complex(kind=8) :: dctmp
  complex(kind=16) :: qctmp
  real(kind = k) :: ini_tmp
  real(kind = k) :: inity_tmp
  real(kind = k) :: nan_tmp
  character(80) :: str
  character(2) :: c
  inf = ieee_value(inf, ieee_positive_inf)
  ninf = ieee_value(inf, ieee_negative_inf)
  nan = ieee_value(inf, ieee_quiet_nan)
  open(8, file="input.txt", action='write',  delim='APOSTROPHE')

  write(unit=8, *) 1.13,1.13,1.13,1.13e11,65536.1,257.1,1.13,(1.11,1.12),(1.13,1.14),(1.15,1.16),inf,ninf,nan
  close(8)

  open(9,file="input.txt", status='OLD', recl=80, delim='APOSTROPHE')

  read(9,*) qtmp,dtmp,rtmp,ltmp,tmp,stmp,btmp,rctmp,dctmp,qctmp,ini_tmp,inity_tmp,nan_tmp
  close(9)
  write(str,*) qtmp
  if(str /= "   1.12999999999999999999999999999999991") STOP 1

  write(str,*) dtmp
  if(str /= "    1.130000000000000") STOP 2

  write(str,*) rtmp
  if(str /= "    1.130000") STOP 3

  write(str,*) ltmp
  if(str /= "             113000000000") STOP 4

  write(str,*) tmp
  if(str /= "        65536") STOP 5

  write(str,*) rctmp
  if(str /= " (1.110000,1.120000)") STOP 6

  write(str,*) dctmp
  if(str /= " (1.130000000000000,1.140000000000000)") STOP 7

  write(str,*) qctmp
  if(str /= " (1.14999999999999999999999999999999992,1.15999999999999999999999999999999993)") STOP 8

  write(str,*) ini_tmp
  if(str /= "                                      Infinity") STOP 9

  write(str,*) inity_tmp
  if(str /= "                                     -Infinity") STOP 10

  write(str,*) nan_tmp
  if(str /= "                                           NaN") STOP 11

  write(*,*) 'PASS'
end program
