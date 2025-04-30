** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Allocatable arrays

	program bq35
	parameter (NT=24)
	common m,mm,n,nn
	data m/1/, mm/3/, n/1/, nn/5/
	integer result(3,8), expect(3,8)
	common /result/result
	data expect /
     +   1, 2, 3, 2, 4, 6, 3, 6,
     +   9, 4, 8, 12, 5, 10, 15, 6,
     +   12, 18, 7, 14, 21, 8, 16, 24 /

	call sub()
	call check(result, expect, NT)
	end
	subroutine sub()
	common m,mm,n,nn
	allocatable ia(:,:)
	allocate (ia(2,2))
	nn = 8
	deallocate (ia)
	allocate (ia(m:mm,n:nn))
	do i = m, mm
	    do j = n, nn
		ia(i,j) = j*i
	    enddo
	enddo
	call pr(ia)
	end
	subroutine pr(ia)
	dimension ia(3,8)
	integer result(3,8)
	common /result/result
	do i = 1, 3
	    do j = 1, 8
		result(i,j) = ia(i,j)
	    enddo
	enddo
	end
