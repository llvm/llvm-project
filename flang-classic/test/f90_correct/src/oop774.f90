! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
logical rslt(1), expect(1)
contains
subroutine foo()
!print *, 'in foo'
end subroutine

subroutine foo2(i)
integer :: i
!print *, 'in foo2', i
rslt(1) = i .eq. -99
end subroutine


end module

subroutine bar()
use mod
block
type t
contains
procedure, nopass :: foo
procedure, nopass :: foo2
generic :: func => foo, foo2 
end type
type(t) :: o

call o%func(-99)
end block
end subroutine

use mod
expect = .true.
rslt = .false.
call bar()
call check(rslt, expect, 1)
end
