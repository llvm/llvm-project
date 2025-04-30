** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Bit manipulation intrinsics: COMPL, AND, OR, NEQV, and EQV

	parameter( N = 30 )
	integer rslts(N), expect(N)
	real xrslts(N)
	logical lrslts(N), t, f
	equivalence (xrslts(2), lrslts(2), rslts(2))
	integer x0,x2
#ifdef __PGLLVM__
	volatile x0,i1
#endif

	data i1, i2, i3, in2, i5, x0, x2, t, f/
     +        1,  2,  3, -2,   5,  0, 2, .true., .false./

	data expect / -1, -2, 'FFFF0000'x, -1, 0, 1,
     +                'F00'x, 1, 1, 2, 4, -1,
     +                'ff0F'x, 2, 1, 4, -1, 2,
     +                'f00f'x, 6, 3, 1, 4, 5,
     +                'ffff0ff0'x, -7, 4, -2, 1, 2            /

c  tests 1 - 6: NOT

	rslts(1) = NOT(0)
	rslts(2) = NOT(i1)
	rslts(3) = NOT('FFFF'x)
	xrslts(4) = COMPL (x0)
	lrslts(5) = NOT(- i1) .or. f
	rslts(6) = COMPL(x0) + i2

c  tests 7 - 12: AND

	rslts(7) = and('F0F'x, 'FF00'x)
	rslts(8) = and(i1, i3)
	rslts(9) = and(7, 3 * i3)
	rslts(10) = and(x2, -1)
	rslts(11) = and('FFFFF'x, x2) + i2
	lrslts(12) = and(.true., t)

c  tests 13 - 18: OR

	rslts(13) = OR('ff00'x, '0f0f'x)
	rslts(14) = OR(x2, 0)
	rslts(15) = NOT(OR(in2, i2))
	rslts(16) = i3 - OR(t, f)
	rslts(OR(17, i2-i1)) = OR(COMPL(x0), AND(i3, .true.))
	xrslts(18) = OR(i2, i2)

c  tests 19 - 24: NEQV

	rslts(19) = NEQV('FF00'x, '0f0f'x)
	rslts(20) = neqv(i3, i5)
	if (neqv(i3, 2))  rslts(21) = i3 - neqv(x2, 2)
	xrslts(22) = neqv( or(i1, 1), and(1, f) )
	rslts(23) = neqv(0, i1) + i3
	rslts(24) = i3 - neqv(i1, -1)

c  tests 25 - 30: EQV

	rslts(25) = EQV('FF00'x, '0f0f'x)
	rslts(26) = eqv(i3, i5)
	if (eqv(i3, 3))  rslts(27) = i3 - eqv(x2, 2)
	xrslts(28) = eqv( or(i1, 1), and(1, f) )
	rslts(29) = eqv(0, i1) + i3
	rslts(30) = i3 - eqv(i1, -1)

c  check results:

	call check(rslts, expect, N)
	end
