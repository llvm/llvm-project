! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module mod
  logical rslt(2), expect(2)
  integer y
  abstract interface
     subroutine bak(x)
       integer x
     end subroutine bak
  end interface
contains
  subroutine foo()
    procedure(bar), pointer :: p
    integer a
    p=>bar
    a = 0
    call bar(200)
    rslt(1) = a .eq. 200
!   print *, a
    a = 1 
    call p(300)
    rslt(2) = a .eq. 300
!   print *, a
    
  contains
    
    subroutine bar(x)
      integer x
      a = x
!     print *, a
    end subroutine bar
  end subroutine foo
end module mod

use mod
expect = .true.
rslt = .false.
call foo()
call check(rslt, expect, 2)

end program
