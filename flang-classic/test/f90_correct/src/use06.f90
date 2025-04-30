!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM
! Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for USE statement when there is rename of user-defined operator in the
! ONLY option.

module m
  interface operator(.add.)
    module procedure func
  end interface
contains
  integer function func(x, y)
    integer, intent(in) :: x, y
    func = x + y
  end function
end module

program p
  use m, only: operator(.localadd.) => operator(.add.)
  implicit none
  if ((1 .localadd. 2) /= 3) STOP 1
  print *, "PASS"
end program
