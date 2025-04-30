** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Shift intrinsics: LSHIFT, RSHIFT

	program shift
	parameter(N = 18)
	integer result(N), expect(N)
	integer*4 xresult(N)
	logical lresult(N), t, f
	equivalence(result(N), xresult(N), lresult(N))
	integer SB, x2
	parameter(SB = '80000000'x)

	data i2,  x2,      t,       f, i1,           m, in31
     +      / 2,   2, .true., .false.,  1, 'FFFFFFFF'x,  -31 /

	data expect /4, 4, SB, -2, 'f0'x, 12345, '00000004'x,
     +               1, 'F0F'x, '7fffffff'x, 1, SB,
     +               '3ffffffc'x, 'f0'x, '20000'x, SB, 1, 0        /

c  --- tests 1 - 6: LSHIFT

	result(1) = LSHIFT(1, 2)
	result(2) = lshift(x2, i2 - 1)
	result(3) = lshift(i1, 31)
	lresult(4) = lshift(-1, 1)
	xresult(5) = lshift('F0'x, 0)
	result(6) = lshift(12345, 2 - i2)

c  --- tests 7 - 12: RSHIFT

	result(7) = rshift(4, 1) + rshift(i2, 0)
	result(8) = rshift(-1, 31)
	result(9) = rshift('F0F0'x, i2*2)
	xresult(10) = rshift(-i2, '1'x)
	lresult(11) = rshift(x2, i2 - i1)
	result(12) = rshift(f .or. t, - (i1 - i2)) + i1

c  --- tests 13 - 18: combinations of bit operations

	result(13) = rshift( lshift(m, 4), 2)
	result(14) = lshift( and(m, 'f'x), 4)
	result(15) = and (lshift(1, 17), m)
	result(16) = lshift(1, -in31)
	result(17) = rshift(SB, -in31)
	result(18) = or( lshift(2, -in31), rshift(2, i2))

c  --- check results:

	call check(result, expect, N)
	end
