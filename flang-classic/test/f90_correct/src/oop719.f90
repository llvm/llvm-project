! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Compile with -Mallocatable=03 to exercise the bug fix (disregard if 
! -Mallocatable=03 is now the default behavior in the compiler).

module mod
  integer count
  type base
     integer :: x
  end type base
  
  type comp
     class(base), allocatable :: b
   contains
     final :: dtor
  end type comp
  
contains
  
  subroutine dtor(this)
    type(comp) :: this
    count = count + 1
  end subroutine dtor
  
  subroutine foo()
    type(comp) :: x, y
    
    allocate(y%b)
    y%b%x = 99
    
    ! final subroutine, dtor, gets called 3 times (called for x,y, and  
    ! the temp used in the x = y assignment below).
    
    x = y
  end subroutine foo
  
end module mod

use mod
logical rslt(1), expect(1)
count = 0
call foo()

expect = .true.
rslt(1) = count .eq. 3
call check(rslt, expect, 1)
end
