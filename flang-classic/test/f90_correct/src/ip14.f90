!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! from iso_varying_string test, make sure allocatables and pointers
! deallocated in contained subprograms work

program p

integer,dimension(:),allocatable :: ar
integer,dimension(:),pointer :: br

integer,dimension(6) :: result,expect
data expect/2,5,40,2,5,40/

allocate(ar(1:10))
allocate(br(1:10))
ar = 5
br = 5
call sub()

result(1) = lbound(ar,1)
result(2) = ubound(ar,1)
result(3) = sum(ar)
result(4) = lbound(br,1)
result(5) = ubound(br,1)
result(6) = sum(br)

call check(result,expect,6)

contains

subroutine sub

 deallocate(ar)
 allocate(ar(2:5))
 ar = 10
 deallocate(br)
 allocate(br(2:5))
 br = 10
end subroutine
end
