** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*--- Induction uses, removal of initval def bug
*    

	program p
	parameter (N=2)
	integer i, j, result(N), expect(N)
	common i, j, result, expect

	data expect /
     +   11, 1
     +  /

	result(1) = 99
	result(2) = 99
	call sub(10, result)
	call sub(0, result)
	call check(result, expect, N)
	end
	subroutine sub(n, ir)
	dimension ir(*)
	do i = 1, n	! can't delete init def of i (i = 1)
	enddo
	call foo(i, ir)
	end
	subroutine foo(i, ir)
	dimension ir(*)
	common /k/k
	data k /1/

	ir(k) = i
	k = k + 1
	end
