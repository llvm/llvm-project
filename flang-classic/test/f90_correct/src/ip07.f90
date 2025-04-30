!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Host array parameter used by an internal subprogram
!
program test
integer, dimension(20) :: result, expect
call z(result, 20)
data expect/1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5/
call check(result, expect, 20)
end

subroutine z(result, n)
integer result(n)
integer, dimension(5), parameter :: idxatt = (/ 1, 2, 3, 4, 5 /)
integer, dimension(5) :: jjj
    jjj = idxatt
    result(1:5) = idxatt
    call zsub1
    contains
	subroutine zsub1
	result(6:10)  = idxatt
	result(11:15) = jjj
	call zsub2(idxatt)
	endsubroutine
	subroutine zsub2(ii)
	integer ii(:)
	result(16:20) = ii
	endsubroutine
end
