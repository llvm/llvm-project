! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test I/O of NML with quad-precision data

program test
  character(80) :: str
  integer, parameter :: k = 16
  real(kind = k) :: tmpa = 1.123456789123456789123456789123456789_16
  real(kind = k) :: tmpb(3) = [1111.123456789123456789123456789123456789_16,1.22222222222222_16,1.33333333333333333333333333_16]
  integer :: rslt, expct

  type t0
  real(kind = k) :: tmpa
  real(kind = k) :: tmpb
  end type

  type t1
  real(kind = k) :: tmpa
  real(kind = k) :: tmpb
  end type

  type(t0) tn
  type(t1) tk

  rslt = 0
  expct = -1
  namelist /tdata/ tmpa, tmpb, tn, tk
  tn%tmpa = 1.987654321987654321987654321987654321_16
  tn%tmpb = 2.987654321987654321987654321987654321_16
  tk%tmpa = 1.987654321987654321987654321987654321_16
  tk%tmpb = 2.987654321987654321987654321987654321_16

  open(8,file="tnmlist.in", action='write',  delim='APOSTROPHE')

  write(unit=8,nml=tdata)
  close(8)

  tmpa = 0.0
  tmpb(1:3) = [0.0,0.0,0.0]
  tn%tmpa = 0.0
  tn%tmpb = 0.0
  tk%tmpa = 0.0
  tk%tmpb = 0.0

  open(9,file="tnmlist.in", status='OLD', recl=80, delim='APOSTROPHE')
  read(9,nml=tdata)
  close(9)

  write(str,100) tmpa
  if(str /= "   0.112345678912345678912345678912345676E+01") STOP 1
  write(str,100) tmpb(1)
  if(str /= "   0.111112345678912345678912345678912345E+04") STOP 2
  write(str,100) tmpb(2)
  if(str /= "   0.122222222222221999999999999999999997E+01") STOP 3
  write(str,100) tmpb(3)
  if(str /= "   0.133333333333333333333333332999999997E+01") STOP 4
  write(*,*) 'PASS'
  100 FORMAT('',e45.36)

end program test
