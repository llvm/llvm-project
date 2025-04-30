** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   PARAMETER statements for character string constants,
*   including constant folding of concatenate operation.

	parameter (jlen = 3)
	implicit character*(*) (a-b), character*6 (d-e, r, j)
	parameter(a = 'x', b = 'functions')

	parameter(dcon = '123456')
	character a2*2, a3*(jlen), aa*(*), a4*4, d1
	parameter(a2 = 'xy', a3 = 'zzz', aa = a2 // a3)

c   --- definitions requiring truncation or padding of string:
	parameter(a4 = 'letters', d1 = 'first' // 'last')
	parameter(dx = 'a', dy = 'abc' // ('d' //'e') )

	parameter(alpha = "abcdefghijklmnopqrstuvwxyz",
     +            blong = alpha // alpha // alpha // alpha)

	character*9 cval, cval2*104

	integer rslts(16), expect(16)

c  --------- tests 1 - 5:

	rslts(1) = jlen
	rslts(2) = len(a)
	rslts(3) = ichar(a)
	data cval /b/
	rslts(4) = ichar( cval(9:) )
	rslts(5) = len(b)

c  --------- tests 6 - 10:

	rslts(6) = len(dcon)
	rslts(7) = and(1, dcon .eq. '123456')
	rslts(8) = len(a2) + len(a3)
	rslts(9) = len(aa)
	rslts(10) = and(1, 'xyzzz' .ne. aa)

c  --------- tests 11 - 16:

	rslts(11) = len(a4) + and(1, a4 .eq. 'lett')
	rslts(12) = ichar(d1) * len(d1)
	rslts(13) = len(dx) * and(1, dx .eq. 'a     ')
	if (dy .eq. 'abcde ')   rslts(14) = 4
	rslts(15) = len(blong)
	data cval2 / blong /
	rslts(16) = ichar( cval2(104:104) )

c  ---------- check results:

	call check(rslts, expect, 16)
	data expect / 3, 1, '170'O, '163'o, 9,
     +                6, 1, 5, 5, 0,
     +                5, '146'o, 6, 4, 104, '172'o  /
	end
