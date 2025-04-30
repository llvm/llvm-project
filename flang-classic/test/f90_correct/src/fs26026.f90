! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module m
implicit none
integer :: i = 0
contains
  subroutine p()
    i = 1
  end subroutine p

  subroutine foo(fun_ptr)
    procedure(p), pointer, intent(out) :: fun_ptr
    fun_ptr => p
  end subroutine
end module m

program test
use m
implicit none
procedure(), pointer :: x
call foo(x)
call x()
if (i == 1) then
  write(*,*) 'PASS'
else
  write(*,*) 'FAIL'
  STOP 1
end if
end program test

