** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   KANJI - NCHARACTER type in common blocks and EQUIVALENCE statements.

	parameter (N=11)
	integer expect(N), rslts(N)

	integer*1 i
	ncharacter a*11, bb(8)*1, eee*7
	ncharacter d(5,6)*3, e*1, ee(5)*2
	integer*2  ss(4)

	common i, a
	common /ccc/ bb

	equivalence (e, d(2, 3)(3:) )
	equivalence (ee(1)(2:), a(3:3) )
	equivalence (a(6:6), ss)
	equivalence (bb(8), eee(4:4))

	data d(2, 3) / nc'789' /

	data expect /	7, 55, 57,
     +			66, 75, 70,
     +			98, 97, 1011, 66, 'FFFF'x
     +			/

	rslts(1) = i 			! 7
	rslts(2) = ichar(d(2,3)(1:1))	! '7'
	rslts(3) = ichar(e)		! '9'

	rslts(4) = ichar(ee(1)(1:1))	! 'B'
	rslts(5) = ichar(ee(5)(2:2))	! 'K'
	rslts(6) = ss(1)		! 'F'

	rslts(7) = ichar(eee(1:1))	! 'b'
	rslts(8) = ichar( bb(3) )	! 'a'
	rslts(9) = ichar(eee(2:2))	! 1011
	rslts(10) = ichar(eee(7:7))	! 66
	rslts(11) = ichar(eee(3:3))	! 'FFFF'x

	call check(rslts, expect, N)
	end


	blockdata
	common i, a			! 24 bytes
	common /ccc/ b, ss		! 22 bytes

	ncharacter a*11, b(5)*1
	integer*1 i
	integer*2 ss(6)

	data i /7/, a / nc'ABCDEFGHIJK' /
	data b / 3*nc'a', 2*nc'b'/,  ss / 1011, -1, 1033, 44, 1055, 66/
	end
