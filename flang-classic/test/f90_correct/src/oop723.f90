! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! tests internal procedure as a pointer target

module mod
  logical rslt(2), expect(2)

contains

  subroutine foo()
    procedure(bar), pointer :: p
    integer a
    p=>bar
    a = 0
    call bar()
    !print *, a
    rslt(1) = a .eq. -99
    a = 1
    call p()
    !print *, a
    rslt(2) = a .eq. -99
    
  contains
    
    subroutine bar()
      a=-99
    end subroutine bar
  end subroutine foo
end module mod

use mod
expect = .true.
rslt = .false.
call foo()
call check(rslt, expect, 2)
end program
