! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod

logical rslt(1), expect(1)
type obj
contains
procedure :: bar
end type 

contains
subroutine foo()
rslt(1) = .true.
end subroutine

subroutine bar(this)
class(obj) :: this

block
type t
contains
procedure, nopass :: foo
end type
type(t) :: o

call o%foo()
end block
end subroutine

end module


use mod
type(obj) :: x
expect = .true.
rslt = .false.
call x%bar()
call check(rslt, expect, 1)
end
