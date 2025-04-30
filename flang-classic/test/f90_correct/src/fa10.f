** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Typeless shift intrinsic: SHIFT.

	program shift_
	parameter(N = 18)
	integer result(N), expect(N)
	real xresult(N), x2
	logical lresult(N), t, f
	equivalence(result(N), xresult(N), lresult(N))
	integer xx2
	equivalence(x2,xx2)
	integer SB
	parameter(SB = '80000000'x)

	data i2,  x2,      t,       f, i1,           m, in31
c     +      / 2, 2.0, .true., .false.,  1, 'FFFFFFFF'x,  -31 /
     +      / 2, 2.0,     1,       0,  1, 'FFFFFFFF'x,  -31 /

	data expect /4, 2 * SB, -2, 'f0'x, 2,
     +               4, 1, 'F0F'x, '7fffffff'x, '20000000'x, 1,
     +               '3ffffffc'x, 'f0'x, '20000'x, SB, 1, 0     /

c  --- tests 1 - 6: LSHFT

	result(1) = SHIFT(1, 2)
	result(2) = shift(xx2, i2 - 1)
	result(3) = shift(i1, 31)
	lresult(4) = shift(-1, 1)
	xresult(5) = shift('F0'x, 0)
	result(6) = shift(2, 2 - i2)

c  --- tests 7 - 12: RSHFT

	result(7) = shift(4, -1) + shift(i2, 0)
	result(8) = shift(-1, -31)
	result(9) = shift('F0F0'x, -i2*2)
	xresult(10) = shift(-i2, 'ffffffff'x)
	lresult(11) = shift(xx2, i1 - i2)
	result(12) = shift(f .or. t, - (i2 - i1)) + i1

c  --- tests 13 - 18: combinations of bit operations

	result(13) = shift( shift(m, 4), -2)
	result(14) = shift( and(m, 'f'x), 4)
	result(15) = and (shift(1, 17), m)
	result(16) = shift(1, -in31)
	result(17) = shift(SB, in31)
	result(18) = or( shift(2, -in31), shift(2, -i2))

c  --- check results:

	call check(result, expect, N)
	end
