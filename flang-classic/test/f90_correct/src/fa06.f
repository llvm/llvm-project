** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Milstd & VMS intrinsics: ISHFTC, MVBITS, IBITS, BTEST, IBSET, IBCLR

C    test both generic and intrinsic forms.

	program fa06
	parameter n = 23
	integer results(n), expect(n)
	integer*2 i1010, i1100, ii
	integer*4 j1010, j1100, jj
	integer*4 ibitslen0
	data i1010, i1100, j1010, j1100
     +	    / '1010'x, '1100'x, '1010'x, '1100'x/

	results(1) = 0
	if (btest(i1010, 12)) results(1) = 1
	results(2) = 0
	if (btest(j1010, 12)) results(2) = 1
	results(3) = 0
	if (bitest(i1100, 8)) results(3) = 1
	results(4) = 0
	if (bjtest(j1100, 8)) results(4) = 1

	results(5) = ibset(i1010, 5)
	results(6) = ibset(j1010, 6)
	results(7) = iibset(i1100, 9)
	results(8) = jibset(j1100, 13)

	results(9) = ibclr(i1010, 12)
	results(10) = ibclr(j1010, 12)
	results(11) = iibclr(i1100, 8)
	results(12) = jibclr(j1100, 8)

	results(13) = ibits(i1010, 4, 4)
	results(14) = ibits(j1010, 12, 12)
	results(15) = iibits(i1100, 8, 12)
	results(16) = jibits(j1100, 8, 12)

	ii = i1100
	call mvbits(i1100, 8, 8, ii, 0)
	results(17) = ii
	jj = j1010
	call mvbits(j1100, 8, 11, jj, 8)
	results(18) = jj

	results(19) = ishftc(i1100, 1, 9)
	results(20) = ishftc(j1010, 8, 12)
	results(21) = iishftc(i1010, 1, 5)
	results(22) = jishftc(j1100, 1, 13)

        results(23) = ibitslen0(7,0)

	call check(results, expect, n)
	data expect / 1, 1, 1, 1,		! btest
     +    '1030'x, '1050'x, '1300'x, '3100'x,   ! bset
     +    '10'x, '10'x, '1000'x, '1000'x,	! ibclr
     +    1, 1, '11'x, '11'x,			! ibits
     +    '1111'x, '1110'x,			! mvbits
     +    '1001'x, '1001'x, '1001'x, '0201'x,	! ishftc
     +    0                                     ! ibits, LEN=0
     +  /
	end
	integer*4 function ibitslen0(iv, len)
	integer*4 iv, len
	ibitslen0 = ibits(iv, 1, len)
	end
