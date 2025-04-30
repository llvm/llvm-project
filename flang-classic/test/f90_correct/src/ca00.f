** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Assumed size arrays.

	program p
	integer rslts(7), expect(7), f

	real a(2,2,2,2,2,2,10:11)
	common b(3:5, -5:-2)

	call sub(rslts, a)
	rslts(2) = a(1,1,1,1,1,1,10)
	rslts(3) = a(2,2,2,2,1,2,11)

	b(2, -4) = 3.1
	rslts(4) = f(3,5,b)
	rslts(5) = b(4, -4)
	rslts(6) = b(3, -2)

	call check(rslts, expect, 7)
	data expect / 2, 8, -1, 6, -2, 4, 4 /
	end
c--------------
	subroutine sub(ar, aa)
	dimension ar(*)
	integer ar
	real aa(2,2,2,2,2,2,10:*)

	data i1, i11 / 1, 11 /

	ar(1) = 2
	ar(7) =  4

	aa(1,1,1,1,1,1,10) = 8.0
	aa(2,2,2,2,i1,2,i11) = -1.0

	return
	end
c--------------
	integer function f(n, m, b)
	real b(n:m, -m:*)

	b(4, -4) = -2.1
	b(3, -2) = 4.01
	f = b(2, -4) + b(2, -4)
	end
