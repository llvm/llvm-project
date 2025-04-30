** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Miscellaneous intrinsics which may appear in parameter statements
*   used in non-parameter context

	program ff01
	complex c1, c2
	double complex cd1, cd2
	double precision d1, d2, d3, d4, d5, d6, d7, d8, d9
	character ch
	logical l1, l2, l3, l4

	parameter (N=40)

	integer result(N)
	integer expect(N)
	common result, expect
	
	i1 = abs(2.5)	! tests 1 - 2
	i2 = abs(-2.5)
	
	x1 = abs(2.5)	! tests 3 - 6
	x2 = abs(-4.5e+3)
	d1 = abs(2.5)
	d2 = abs(-5.4999d+4)
	i3 = nint(x1)
	i4 = nint(-x2)
	i5 = idnint(d1)
	i6 = idnint(-d2)
	
	c1 = (2.54, -4.89)	! tests 7-9
	c2 = conjg(c1)
	x3 = aimag(c2)
	
	cd1 = (4.49, -10.2)	! tests 10-12
	cd2 = conjg(c2)
	d3 = dimag(cd2)

	d4 = dprod(x1, x2)	! test 13

	i7 = max(i1, i2, i3, i4, i5, i6)	! tests 14-15
	i8 = min(i1, i2, i3, i4, i5, i6, i7)

	x4 = max(x1, x2, x3)			! tests 16-17
	x5 = min(x1, x2, x3, x4)

	d5 = max(d1, d2, d3, d4)		! tests 18-19
	d6 = min(d1, d2, d3, d4, d5)

	i9 = iand('aaaaaaaa'x, '84848484'x)	! tests 20-23
	i10 = ior('78787878'x, '87878787'x)
	i11 = not('f0f0f0f0'x)
	i12 = ieor('bbbbbbbb'x, '8af8afcd'x)

	i13 = mod(13, -3)			! tests 24-26
	x6 = mod(-13.3, 5.0)
	d7 = mod(15.3d+2, 1.53d+2)

	i14 = ichar('a')			! test 27

	i15 = dim(20,15)			! tests 28-33
	i16 = dim(1,1)
	x7 = dim(20.0,15.0)
	x8 = dim(1.0,1.0)
	d8 = dim(20.0d+1, 15.0d+1)
	d9 = dim(1.0d+1, 1.0d+1)

	i17 = ishft('a0000000'x, 1)		! tests 34-35
	i18 = ishft('80000001'x, -1)

	ch = char(32)			! test 36

	l1 = lge('abcd', 'abcd')		! tests 37-40
	l2 = lgt('abcd', 'abc')
	l3 = lle('abc', 'abc')
	l4 = llt('abc', 'abcd')

	result(1) = i1
	result(2) = i2

	result(3) = i3
	result(4) = i4
	result(5) = i5
	result(6) = i6

	result(7) = c1
	result(8) = x3
	result(9) = cd1

	result(10) = cd1
	result(11) = d3
	result(12) = cd2

	result(13) = d4

	result(14) = i7
	result(15) = i8

	result(16) = x4
	result(17) = x5

	result(18) = d5
	result(19) = d6

	result(20) = i9
	result(21) = i10
	result(22) = i11
	result(23) = i12

	result(24) = i13
	result(25) = x6
	result(26) = d7

	result(27) = i14

	result(28) = i15
	result(29) = i16
	result(30) = x7
	result(31) = x8
	result(32) = d8
	result(33) = d9

	result(34) = i17
	result(35) = i18

	result(36) = ichar(ch)

	result(37) = iand(l1, 1)
	result(38) = iand(l2, 1)
	result(39) = iand(l3, 1)
	result(40) = iand(l4, 1)

	call check(result, expect, N)
	data result/N*-1/
	data expect/
     + 2, 2,					! tests 1 - 2
     + 3, -4500, 3, -54999,			! tests 3 - 6
     + 2, 4, 4,					! tests 7-9
     + 4, -4, 2,				! tests 10-12
     + 11250,					! test 13
     + 3, -54999,				! tests 14-15
     + 4500, 2,					! tests 16-17
     + 54999, -4,				! tests 18-19
     + -2139062144, -1, 252645135, 826479734,	! tests 20-23
     + 1, -3, 0,				! tests 24-26
     + 97,					! test 27
     + 5, 0, 5, 0, 50, 0,			! tests 28-33
     + 1073741824, 1073741824,			! tests 34-35
     + 32,					! test 36
     + 1, 1, 1, 1 /				! tests 37-40
	end
