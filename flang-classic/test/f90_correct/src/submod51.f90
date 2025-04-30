! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module A
implicit none
integer, target :: c
interface
  module subroutine A1(i)
    integer, intent(inout) :: i
  end subroutine A1

  module subroutine sub(ptr)
    integer, pointer :: ptr
  end subroutine
end interface

integer :: incr 

contains
  module subroutine A1(i)
    integer, intent(inout) :: i
    incr = incr + 1
    !print *, incr !<- should print 2
    if (incr .eq. 2) then
      print *, "PASS"
    else
      print *, "FAIL"
    endif
  end subroutine A1

  module procedure sub
    ptr => c
  end procedure
end module

program test
use A
integer :: i = 1
integer, pointer :: toC
incr = 1
call A1(i)
call sub(toC)
end
