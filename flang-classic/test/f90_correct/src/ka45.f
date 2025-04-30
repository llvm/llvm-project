** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*--- Induction linear combinations
*    

	program p
	parameter (N=4)
	integer i, j, result(N, N), expect(N, N)
	common i, j, result, expect

	data expect /
     +   1, 11, 31, 32,		! t1, t2, t4, t4
     +  33,  2, 12, 41,		! t4, t1, t2, t5
     +  42, 43,  3, 13,		! t5, t5, t1, t2
     +  21, 22, 23,  4		! t3, t3, t3, t1
     +  /

	call t1(result, 4)

	call t2(result, 4)

	call t3(result, 4)

	call t4(result(3,1), 4)

	call t5(result(4,2), 4, 1, 0)

	call check(result, expect, N*N)
	end
	subroutine t1 (ir, n)	! (1,1),(2,2),(3,3),(4,4)
	dimension ir(n, n)
	do i = 1, n
	   ir(i,i) = i
	enddo
	end
	subroutine t2 (ir, n)	! (2,1),(3,2),(4,3)
	dimension ir(n, n)
	do i = 1, n-1
	   ir(i+1, i) = i + 10
	enddo
	end
	subroutine t3 (ir, n)	! (1,4), (2,4), (3,4)
	dimension ir(n, n)
	do i = 1, n-1
	   ir(-(3*i - 4*i), 4) = i + 20
	enddo
	end
	subroutine t4 (ir, n)	! (3,1),(4,1),(1,2)
	dimension ir(n, *)
	do i = 1, n-1
	   ir((-(3*i)) + (4 * i), 1) = i + 30
	enddo
	end
	subroutine t5(ir, n, m1, m2)	! (4,2),(1,3),(2,3)
	dimension ir(n, *)
	do i = 1, n-1
	    ir(m1*i + m2*i, 1) = i + 40
	enddo
	end
