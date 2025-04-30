! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! tests internal procedures as pointer targets 

module mod
  logical rslt(3), expect(3)
contains
  
  subroutine parent(p)
    procedure(), pointer :: p
    call p()
  end subroutine parent
  
  subroutine foo()
    procedure(bar), pointer :: p
    integer a
    a=0
    p=>bar
    call bar()
    rslt(1) = a .eq. 1
    call p()
    rslt(2) = a .eq. 2
    call parent(p)
    rslt(3) = a .eq. 3
  contains
    
    subroutine bar()
      a=a+1
    end subroutine bar
  end subroutine foo
end module mod

use mod
expect = .true.
rslt = .false.
call foo()
call check(rslt,expect,3)
end program
