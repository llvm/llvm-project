** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Optimizer bug: flowgraph altered due to zerotrip do statement

	program ka20
	parameter (N=1)
	integer result(N), expect(N)
	data expect/11/
	integer a(100),b(100)
	integer zloop1
	a = 0
	b = 0
	result(1) = zloop1(a,b,100)
	call check(result, expect, N)
	end
	integer function zloop1(a,b,n)
	integer a(n), b(n)
	call copy(0, m)
	it = 11			! should have a use after next do loop
	do i = 1, m		! loop isn't executed
	    it = a(i) + 1
	    b(i) = it + 1
	enddo
	zloop1 = it
	end
	subroutine copy(n,m)
	m = n
	end
