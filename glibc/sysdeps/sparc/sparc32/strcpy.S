/* Copy SRC to DEST returning DEST.
   For SPARC v7.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jj@ultra.linux.cz>.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

	/* Normally, this uses ((xword - 0x01010101) & 0x80808080) test
	   to find out if any byte in xword could be zero. This is fast, but
	   also gives false alarm for any byte in range 0x81-0xff. It does
	   not matter for correctness, as if this test tells us there could
	   be some zero byte, we check it byte by byte, but if bytes with
	   high bits set are common in the strings, then this will give poor
	   performance. You can #define EIGHTBIT_NOT_RARE and the algorithm
	   will use one tick slower, but more precise test
	   ((xword - 0x01010101) & (~xword) & 0x80808080),
	   which does not give any false alarms (but if some bits are set,
	   one cannot assume from it which bytes are zero and which are not).
	   It is yet to be measured, what is the correct default for glibc
	   in these days for an average user.
	 */

	.text
	.align		4

ENTRY(strcpy)
	mov		%o0, %g2
	andcc		%o1, 3, %g0
	be		10f
	 sethi		%hi(0x80808080), %o4

	ldub		[%o1], %o5
	stb		%o5, [%o0]
	cmp		%o5, 0
	add		%o0, 1, %o0
	be		0f
	 add		%o1, 1, %o1
	andcc		%o1, 3, %g0
	be		4f
	 or		%o4, %lo(0x80808080), %o3
	ldub		[%o1], %o5
	stb		%o5, [%o0]
	cmp		%o5, 0
	add		%o0, 1, %o0
	be		0f
	 add		%o1, 1, %o1
	andcc		%o1, 3, %g0
	be		5f
	 sethi		%hi(0x01010101), %o4
	ldub		[%o1], %o5
	stb		%o5, [%o0]
	cmp		%o5, 0
	add		%o0, 1, %o0
	be		0f
	 add		%o1, 1, %o1
	b		6f
	 andcc		%o0, 3, %g3

10:	or		%o4, %lo(0x80808080), %o3
4:	sethi		%hi(0x01010101), %o4
5:	andcc		%o0, 3, %g3
6:	bne		10f
	 or		%o4, %lo(0x01010101), %o2
1:	ld		[%o1], %o5
	add		%o1, 4, %o1
	sub		%o5, %o2, %o4
	add		%o0, 4, %o0
	andcc		%o4, %o3, %g0
	be,a		1b
	 st		%o5, [%o0 - 4]

	srl		%o5, 24, %g5
	andcc		%g5, 0xff, %g0
	be		1f
	 srl		%o5, 16, %g5
	andcc		%g5, 0xff, %g0
	be		2f
	 srl		%o5, 8, %g5
	andcc		%g5, 0xff, %g0
	be		3f
	 andcc		%o5, 0xff, %g0
	bne		1b
	 st		%o5, [%o0 - 4]
	retl
	 mov		%g2, %o0
3:	srl		%o5, 16, %o5
	sth		%o5, [%o0 - 4]
	stb		%g0, [%o0 - 2]
	retl
	 mov		%g2, %o0
2:	srl		%o5, 16, %o5
	sth		%o5, [%o0 - 4]
	retl
	 mov		%g2, %o0
1:	stb		%g0, [%o0 - 4]
	retl
	 mov		%g2, %o0

10:	ld		[%o1], %o5
	add		%o1, 4, %o1
	sub		%o5, %o2, %o4
	cmp		%g3, 2
	be		2f
	 cmp		%g3, 3
	be		3f
	 andcc		%o4, %o3, %g0
	bne		5f
	 srl		%o5, 24, %g5
	stb		%g5, [%o0]
	sub		%o0, 1, %o0
	srl		%o5, 8, %g5
	sth		%g5, [%o0 + 2]
1:	add		%o0, 4, %o0
4:	sll		%o5, 24, %g6
	ld		[%o1], %o5
	add		%o1, 4, %o1
	srl		%o5, 8, %g5
	sub		%o5, %o2, %o4
	or		%g5, %g6, %g5
	andcc		%o4, %o3, %g0
	be,a		1b
	 st		%g5, [%o0]
	srl		%o5, 24, %o4
	andcc		%o4, 0xff, %g0
	be		6f
	 srl		%o5, 16, %o4
	andcc		%o4, 0xff, %g0
	be		7f
	 srl		%o5, 8, %o4
	st		%g5, [%o0]
	andcc		%o4, 0xff, %g0
	be		0f
	 andcc		%o5, 0xff, %g0
1:	bne		4b
	 add		%o0, 4, %o0
9:	stb		%g0, [%o0]
0:	retl
	 mov		%g2, %o0
6:	srl		%g5, 16, %g5
	sth		%g5, [%o0]
	retl
	 mov		%g2, %o0
7:	srl		%g5, 16, %g5
	sth		%g5, [%o0]
	stb		%g0, [%o0 + 2]
	retl
	 mov		%g2, %o0
5:	andcc		%g5, 0xff, %g4
	be		9b
	 srl		%o5, 16, %g5
	andcc		%g5, 0xff, %g0
	be		7f
	 srl		%o5, 8, %g5
	andcc		%g5, 0xff, %g0
	stb		%g4, [%o0]
	sth		%g5, [%o0 + 1]
	sub		%o0, 1, %o0
	bne		1b
	 andcc		%o5, 0xff, %g0
	retl
	 mov		%g2, %o0
7:	stb		%g4, [%o0]
	stb		%g0, [%o0 + 1]
	retl
	 mov		%g2, %o0

2:	andcc		%o4, %o3, %g0
	bne		5f
	 srl		%o5, 16, %g5
	sth		%g5, [%o0]
	sub		%o0, 2, %o0
1:	add		%o0, 4, %o0
4:	sll		%o5, 16, %g6
	ld		[%o1], %o5
	add		%o1, 4, %o1
	srl		%o5, 16, %g5
	sub		%o5, %o2, %o4
	or		%g5, %g6, %g5
	andcc		%o4, %o3, %g0
	be,a		1b
	 st		%g5, [%o0]
	srl		%o5, 24, %o4
	andcc		%o4, 0xff, %g0
	be		7f
	 srl		%o5, 16, %o4
	st		%g5, [%o0]
	andcc		%o4, 0xff, %g0
	be		0b
	 srl		%o5, 8, %o4
1:	andcc		%o4, 0xff, %g0
	be		8f
	 andcc		%o5, 0xff, %g0
	bne		4b
	 add		%o0, 4, %o0
	sth		%o5, [%o0]
	retl
	 mov		%g2, %o0
7:	srl		%g5, 16, %g5
	sth		%g5, [%o0]
	stb		%g0, [%o0 + 2]
	retl
	 mov		%g2, %o0
8:	stb		%g0, [%o0 + 4]
	retl
	 mov		%g2, %o0
5:	srl		%o5, 24, %g5
	andcc		%g5, 0xff, %g0
	be		9b
	 srl		%o5, 16, %g5
	andcc		%g5, 0xff, %g0
	sth		%g5, [%o0]
	sub		%o0, 2, %o0
	bne		1b
	 srl		%o5, 8, %o4
	retl
	 mov		%g2, %o0

3:	bne		5f
	 srl		%o5, 24, %g5
	stb		%g5, [%o0]
	sub		%o0, 3, %o0
1:	add		%o0, 4, %o0
4:	sll		%o5, 8, %g6
	ld		[%o1], %o5
	add		%o1, 4, %o1
	srl		%o5, 24, %g5
	sub		%o5, %o2, %o4
	or		%g5, %g6, %g5
	andcc		%o4, %o3, %g0
	be		1b
	 st		%g5, [%o0]
	srl		%o5, 24, %o4
	andcc		%o4, 0xff, %g0
	be		0b
	 srl		%o5, 16, %o4
1:	andcc		%o4, 0xff, %g0
	be		8b
	 srl		%o5, 8, %o4
	andcc		%o4, 0xff, %g0
	be		9f
	 andcc		%o5, 0xff, %g0
	bne		4b
	 add		%o0, 4, %o0
	srl		%o5, 8, %o5
	sth		%o5, [%o0]
	stb		%g0, [%o0 + 2]
	retl
	 mov		%g2, %o0
9:	srl		%o5, 8, %o5
	sth		%o5, [%o0 + 4]
	retl
	 mov		%g2, %o0
5:	andcc		%g5, 0xff, %g0
	stb		%g5, [%o0]
	sub		%o0, 3, %o0
	bne		1b
	 srl		%o5, 16, %o4
	retl
	 mov		%g2, %o0
END(strcpy)
libc_hidden_builtin_def (strcpy)
