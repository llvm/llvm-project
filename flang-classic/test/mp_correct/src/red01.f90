!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 
!       OpenMP Reductions
!       Array reduction variables (extension)

module data
integer ir(10,10),nr(10)
integer, parameter :: NTESTS=10
integer,dimension(NTESTS) :: expect = (/10,20,30,40,50,60,70,80,90,100/)
contains
    subroutine init
	do i = 1, 10
	   do j = 1, 10
	       ir(i,j) = j
	   enddo
	   nr(i) = 0
	enddo
    end subroutine
endmodule

program red01
use data
call init
!$omp parallel do reduction(nr)
do i = 1, 10
    nr = nr + ir(i,:)
enddo
!$omp end parallel do
!print *, nr
call check(nr, expect, NTESTS)
end
