** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*--- Induction alias use bug,
*    

	program p
	parameter (N=6)
	integer i, j, result(N), expect(N)
	common i, j, result, expect
	common /k/k
	data k/5/

	data expect /
     +   2, 3, 4, 5, 1, 5
     +  /

	call sub(result, N-1)
	result(6) = k

	call check(result, expect, N)
	end
	subroutine sub(ia, n)
	dimension ia(n)
	common /k/k
	j = 0
	do i = 1, n
	    ia(k) = i
	    j = j + 1	! def between use of k and def of k
	    k = j
	enddo
	end
