** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Adjustable arrays and ENTRY statements.

C   NONSTANDARD:
C      Use of character constants to initialize numeric variable.

	parameter(n = 4)
	integer rslts(n), expect(n)
	character*4 c(3, 2), d(4, 3)
	equivalence (c(2, 1), rslts(3)), (d(1, 1), rslts(4))

	call sub1(rslts, 100, 10)
	call trash
	call e1(11, 12, rslts(2))

	call sub2(c, 3, 2)
	call trash
	call e2(d, 3, 4)

c --- check results:

	data expect / 10, 11, 'abcd', 'x   ' /

	call check(rslts, expect, n)
	end

ccccccccccccccccccccccccccccccccc

	subroutine sub1(a, n, m)
	entry e1(m, n, a)
	integer a(m:n)
	save i
	data i / 10 /

	a(i) = i
	i = i + 1
	end

ccccccccccccccccccccccccc

	subroutine sub2(a, n, m)
	character*(*) a(n, m)
	
	a(n-1, m-1) = 'abcdjksdhfkasdjfalfd'
	return

	entry e2(a, m, n)
	a(1, 1) = 'x'
	end

	subroutine trash()
	integer stk(20)
	do 10 i = 1, 20
	    stk(i) = -100
10	continue
	end
