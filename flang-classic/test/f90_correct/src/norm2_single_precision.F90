!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!


program test
  use ISO_FORTRAN_ENV
  implicit none
  integer, parameter :: expec = 1
  integer :: res

  res = norm2Single()
  call check(res, expec, 1)

contains

function norm2Single()
  implicit none
  integer, parameter :: wp = REAL32
  real, parameter :: tolerance = 1.0E-12
  integer, parameter :: n = Maxexponent(1.0_wp) - 3
  real(wp), parameter :: r = Radix(1.0_wp)
  real(wp), parameter :: big = r**n
  real(wp) :: x5(3) = [big, big, big]
  real(wp) resultSingle
  real(wp), parameter :: expectSingle =  7.36732922E+37_wp
  integer :: norm2Single

  resultSingle = norm2(x5)

  if(abs(resultSingle - expectSingle) < tolerance) then
    norm2Single = 1
    print *, 'Single precision test  match'
  else
    norm2Single = -1
    print *, 'Single precision test  mismatch'
  end if
end function
end program
