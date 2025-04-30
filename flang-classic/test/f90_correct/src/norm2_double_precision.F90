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

  res = norm2Double()
  call check(res, expec, 1)

contains

function norm2Double()
  implicit none
  integer, parameter :: wp = REAL64
  real, parameter :: tolerance = 1.0E-12
  integer, parameter :: n = (Maxexponent(1.0_wp)/2)-1
  real(wp), parameter :: r = Radix(1.0_wp)
  real(wp), parameter :: big = r**n
  real(wp) :: x5(4) = [big, big, big, big]
  real(wp), parameter :: expectDouble = 3.8921198074991259E+307_wp
  real(wp) :: resultDouble
  integer :: norm2Double

  resultDouble = norm2(x5)
  if(abs(resultDouble - expectDouble) < tolerance) then
    norm2Double = 1
    print *, 'Double precision test match'
  else
    norm2Double = -1
    print *, 'Double precision test mismatch'
  end if
end function

end program
