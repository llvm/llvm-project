/* Copy SRC to DEST returning the address of the terminating '\0' in DEST.
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

ENTRY(__stpcpy)
	andcc		%o1, 3, %g0
	be		20f
	 sethi		%hi(0x80808080), %o4

	ldub		[%o1], %o5
	stb		%o5, [%o0]
	cmp		%o5, 0
	add		%o0, 1, %o0
	be		1f
	 add		%o1, 1, %o1
	andcc		%o1, 3, %g0
	be		4f
	 or		%o4, %lo(0x80808080), %o3
	ldub		[%o1], %o5
	stb		%o5, [%o0]
	cmp		%o5, 0
	add		%o0, 1, %o0
	be		1f
	 add		%o1, 1, %o1
	andcc		%o1, 3, %g0
	be		5f
	 sethi		%hi(0x01010101), %o4
	ldub		[%o1], %o5
	stb		%o5, [%o0]
	cmp		%o5, 0
	add		%o0, 1, %o0
	be		1f
	 add		%o1, 1, %o1
	b		6f
	 or		%o4, %lo(0x01010101), %o2
1:	retl
	 add		%o0, -1, %o0

20:	or		%o4, %lo(0x80808080), %o3
4:	sethi		%hi(0x01010101), %o4
5:	or		%o4, %lo(0x01010101), %o2
6:	andcc		%o0, 3, %g0
	bne		16f
	 sub		%g0, 4, %g1

11:	add		%g1, 4, %g1
	ld		[%o1 + %g1], %o5
	sub		%o5, %o2, %o4
#ifdef EIGHTBIT_NOT_RARE
	andn		%o4, %o5, %o4
#endif
	andcc		%o4, %o3, %g0
	be,a		11b
	 st		%o5, [%o0 + %g1]

	/* Check every byte. */
	srl		%o5, 24, %g5
	andcc		%g5, 0xff, %g0
	be		14f
	 srl		%o5, 16, %g5
	andcc		%g5, 0xff, %g0
	be		13f
	 srl		%o5, 8, %g5
	andcc		%g5, 0xff, %g0
	be		12f
	 andcc		%o5, 0xff, %g0
	bne		11b
	 st		%o5, [%o0 + %g1]
	add		%o0, %g1, %o0
	retl
	 add		%o0, 3, %o0
12:	srl		%o5, 16, %o5
	sth		%o5, [%o0 + %g1]
	add		%g1, 2, %g1
	stb		%g0, [%o0 + %g1]
	retl
	 add		%o0, %g1, %o0
13:	srl		%o5, 16, %o5
	sth		%o5, [%o0 + %g1]
	add		%g1, 1, %g1
	retl
	 add		%o0, %g1, %o0
14:	stb		%g0, [%o0 + %g1]
	retl
	 add		%o0, %g1, %o0

15:	srl		%o5, 24, %o4
	srl		%o5, 16, %g1
	stb		%o4, [%o0]
	srl		%o5, 8, %g4
	stb		%g1, [%o0 + 1]
	stb		%g4, [%o0 + 2]
	stb		%o5, [%o0 + 3]
	add		%o0, 4, %o0
16:	ld		[%o1], %o5
	sub		%o5, %o2, %o4
	andcc		%o4, %o3, %g0
	be		15b
	 add		%o1, 4, %o1

	/* Check every byte. */
	srl		%o5, 24, %g5
	andcc		%g5, 0xff, %g4
	be		19f
	 stb		%g4, [%o0]
	srl		%o5, 16, %g5
	andcc		%g5, 0xff, %g4
	be		18f
	 stb		%g4, [%o0 + 1]
	srl		%o5, 8, %g5
	andcc		%g5, 0xff, %g4
	be		17f
 	 stb		%g4, [%o0 + 2]
	andcc		%o5, 0xff, %g4
	stb		%g4, [%o0 + 3]
	bne		16b
	 add		%o0, 4, %o0
	retl
	 sub		%o0, 1, %o0
17:	retl
	 add		%o0, 2, %o0
18:	retl
	 add		%o0, 1, %o0
19:	retl
	 nop
END(__stpcpy)

weak_alias (__stpcpy, stpcpy)
libc_hidden_def (__stpcpy)
libc_hidden_builtin_def (stpcpy)
