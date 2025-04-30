** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Adjustable arrays and ENTRY statements.

C   "Difficult" dimension declarators with respect to entry flow.

	parameter(nt = 4)
	integer ir(nt), expect(nt)
	integer ia(1,1), ib(1,1)
	equivalence (ia, ir(2))
	equivalence (ib, ir(3))
	common in, m, n

	in = 2
	m = 1
	n = 2
	call en1(ir, ir)	! ir(1), ir(4)

	in = 3
	n = 1
	call en2(ia)		! ir(2)

	in = 4
	call en3(ib, 1)		! ir(3)

c --- check results:

	data expect / 1, 3, 4, 2 /

	call check(ir, expect, nt)
	end

ccccccccccccccccccccccccccccccccc

	subroutine en1(ix, iy)
	common in, m, n

	entry en2(iy)
	integer ix(m, m)
	integer iy(n, n)
	integer iz(nn, nn)
	entry en3(iz, nn)

	if (in .ge. 4) then
	    iz(nn,nn) = in
	    return
	endif
	if (in .le. 2) ix(m, m) = 1

	iy(n, n) = in

	return

	end
