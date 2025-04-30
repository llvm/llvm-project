!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM
! Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for rename of user-defined operator in USE statement.

module m
  implicit none
  interface operator(.opr.)
    module procedure :: foo
  end interface
  integer :: opr = 1
contains
  integer function foo(x, y)
    integer, intent(in) :: x, y
    foo = x + y
  end function
end module m

program p
  use m, operator(.localopr.) => operator(.opr.)
  implicit none
  if (opr /= 1) STOP 1
  if ((1 .localopr. 2) /= 3) STOP 2
  print *, "PASS"
end program
