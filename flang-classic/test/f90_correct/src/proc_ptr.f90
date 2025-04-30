!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test the fix for segment fault about procedure pointer when all the following
! three conditions are satisfied:
! 1) module procedue used as proc-interface
! 2) keyword RESULT appears
! 3) none dummy argument

module m
  procedure(g), pointer :: f
contains
  function g() result(z)
    integer :: z(3)
    z = (/1, 2, 3/)
  end function
end module

program test
  use m
  implicit none
  integer :: a(3)
  f => g
  a = f()
  if (.not. all(a .eq. (/1, 2, 3/))) STOP 1

  print *, 'PASS'
end program
