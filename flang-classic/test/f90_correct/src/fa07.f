** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Milstd & VMS intrinsics: ISHFTC, IBITS, BTEST, IBSET, IBCLR

C    test procedural forms.

	program fa07
	parameter n = 20
	intrinsic iishft, jishft
	intrinsic iibits, jibits
	intrinsic iibset, jibset
	intrinsic iibclr, jibclr
	intrinsic iishftc, jishftc
	intrinsic bitest, bjtest
	integer results(n), expect(n)
	integer*2 i1010, i1100, ii, i1, i2, i3
	integer*4 j1010, j1100, ji
	data i1010, i1100, j1010, j1100
     +	    / '1010'x, '1100'x, '1010'x, '1100'x/

	external l2fun, l4fun
	logical*2 l2fun
	logical*4 l4fun
	external i2fun3, i4fun3
	integer*2 i2fun3
	integer*4 i4fun3
	external i2fun4, i4fun4
	integer*2 i2fun4
	integer*4 i4fun4

	results(1) = 0
	i1 = 12		! currently have problems passing constants
	if (l2fun( bitest, i1010, i1)) results(1) = 1
	results(2) = 0
	if (l4fun( bjtest, j1010, 12)) results(2) = 1
	results(3) = 0
	i1 = 8
	if (l2fun(bitest, i1100, i1)) results(3) = 1
	results(4) = 0
	if (l4fun(bjtest, j1100, 8)) results(4) = 1

	i1 = 5
	results(5) = i2fun3(iibset, i1010, i1)
	results(6) = i4fun3(jibset, j1010, 6)
	i1 = 9
	results(7) = i2fun3(iibset, i1100, i1)
	results(8) = i4fun3(jibset, j1100, 13)

	i1 = 12
	results(9) = i2fun3(iibclr, i1010, i1)
	results(10) = i4fun3(jibclr, j1010, 12)
	i1 = 8
	results(11) = i2fun3(iibclr, i1100, i1)
	results(12) = i4fun3(jibclr, j1100, 8)

	i1 = 4
	i2 = 4
	results(13) = i2fun4(iibits, i1010, i1, i1)
	results(14) = i4fun4(jibits, j1010, 12, 12)
	i1 = 12
	i2 = 8
	results(15) = i2fun4(iibits, i1100, i2, i1)
	results(16) = i4fun4(jibits, j1100, 8, 12)

	i1 = 9
	i2 = 1
	results(17) = i2fun4(iishftc, i1100, i2, i1)
	results(18) = i4fun4(jishftc, j1010, 8, 12)
	i1 = 5
	i2 = 1
	results(19) = i2fun4(iishftc, i1010, i2, i1)
	results(20) = i4fun4(jishftc, j1100, 1, 13)

	call check(results, expect, n)
	data expect / 1, 1, 1, 1,		! btest
     +    '1030'x, '1050'x, '1300'x, '3100'x,   ! bset
     +    '10'x, '10'x, '1000'x, '1000'x,	! ibclr
     +    1, 1, '11'x, '11'x,			! ibits
     +    '1001'x, '1001'x, '1001'x, '0201'x	! ishftc
     +  /
	end
	logical*2 function l2fun(a, b, c)
	logical*2 a
	integer*2 b, c
	external a
	l2fun = a(b, c)
	end
	logical*4 function l4fun(a, b, c)
	logical*4 a
	integer*4 b, c
	external a
	l4fun = a(b, c)
	end
	integer*2 function i2fun3(a, b, c)
	integer*2 a, b, c
	external a
	i2fun3 = a(b, c)
	end
	integer*4 function i4fun3(a, b, c)
	integer*4 a, b, c
	external a
	i4fun3 = a(b, c)
	end
	integer*2 function i2fun4(a, b, c, d)
	integer*2 a, b, c, d
	external a
	i2fun4 = a(b, c, d)
	end
	integer*4 function i4fun4(a, b, c, d)
	integer*4 a, b, c, d
	external a
	i4fun4 = a(b, c, d)
	end
