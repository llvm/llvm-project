** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*--- Induction alias uses
*    

	program p
	parameter (N=27)
	integer i, j, result(N), expect(N)
	common i, j, result, expect

	data expect /
     +  1, 2, 3, 4,			! t1 (1-4)
     +  1, 2, 3, 4, 3,			! t2 (5-9)
     +  2, 3, 6, 9,			! t3 (10-13)
     +  2, 3, 6, 9, 3,			! t4 (14-18)
     +  2, 3, 6, 9, 3,			! t5 (19-23)
     +  1, 2, 3, 4			! t6 (24-27)
     +  /

	call t1(result, 4)

	call t2(result(5), 4)

	call t3(result(10), 4)

	call t4(result(14), 4)

	call t5(result(19), 4)

	call t6(result(24), 4)

	call check(result, expect, N)
	end

	subroutine t1(iarr, iup)
	integer iarr(*)
	do i = 1, iup, 2
	    k = i
	    iarr(k) = i
	    iarr(k+1) = i + 1
	enddo
	end

	subroutine t2(iarr, iup)
	integer iarr(*)
	do i = 1, iup, 2
	    k = i
	    iarr(k) = i
	    iarr(k+1) = i + 1
	enddo
	iarr(iup+1) = k
	end

	subroutine t3(iarr, iup)
	integer iarr(*)
	do i = 0, iup - 1, 2
	    k = i + 1
	    iarr(k) = 2*k
	    iarr(k+1) = 3*k
	enddo
	end

	subroutine t4(iarr, iup)
	integer iarr(*)
	do i = 0, iup - 1, 2
	    k = i + 1
	    iarr(k) = 2*k
	    iarr(k+1) = 3*k
	enddo
	iarr(iup + 1) = k
	end

	subroutine t5(iarr, iup)
	integer iarr(*)
	do i = 0, iup - 1, 2
	    k = i + 1
	    iarr(k) = 2*k
	    iarr(k+1) = 3*k
	enddo
	call set(iarr(iup + 1), k)
	end

	subroutine set(ir, iv)
	ir = iv
	end

	subroutine t6(iarr, iup)
	integer iarr(*)
	do i = 0, iup - 1, 2
	    k = i + 1
	    iarr(i + 1) = k
	    iarr(i + 2) = k + 1
	enddo
	end

