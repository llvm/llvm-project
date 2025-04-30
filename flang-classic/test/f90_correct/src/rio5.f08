! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! check that WRITE handles infinities and NaNs correctly after converting such
! values from double precision to quadruple precision

program test
  integer, parameter :: k = 16
  real(kind = 8) :: tmpa = z'7ff0000000000000'
  real(kind = k) :: tmpb
  real(kind = 8) :: tmpc = z'fff0000000000000'
  real(kind = 8) :: tmpd = z'7ff0000000000001'
  real(kind = 8) :: tmpe = z'fff0000000000001'
  character(len = 80) :: str
  logical rslt(4)
  logical expect(4)
  integer::a

  rslt(1) = .true.
  rslt(2) = .true.
  rslt(3) = .true.
  rslt(4) = .true.
  expect(1) = .true.
  expect(2) = .true.
  expect(3) = .true.
  expect(4) = .true.

  tmpb = tmpa
  write(str,*) tmpb
  if (str .ne.  "                                      Infinity") then
    rslt(1) = .false.
  endif

  tmpb = tmpc
  write(str,*) tmpb
  if (str .ne.  "                                     -Infinity") then
    rslt(2) = .false.
  endif

  tmpb = tmpd
  write(str,*) tmpb
  if (str .ne.  "                                           NaN") then
    rslt(3) = .false.
  endif

  tmpb = tmpe
  write(str,*) tmpb
  if (str .ne.  "                                           NaN") then
    rslt(4) = .false.
  endif

  call check(rslt,expect,4)

end program test
