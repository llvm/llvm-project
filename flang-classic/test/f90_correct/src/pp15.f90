! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test noncontiguous pointers.
! Bug when an outer-scope pointer is referenced in an internal procedure
!
integer, parameter:: N=10
integer, target :: a(10,N)
integer, pointer :: p(:)
integer :: expect(N) = (/1,2,3,4,5,6,7,8,9,10/) 
integer :: result(N)

p=>a(2,:)
call foo

!print *,  a(2,:)
result = a(2,1:N)
call check(result, expect, N)

contains
    subroutine foo
    do i = 1, N
        p(i) = i   !p's SDSCNS1 flag must be set in the backend
    enddo
    endsubroutine
end
