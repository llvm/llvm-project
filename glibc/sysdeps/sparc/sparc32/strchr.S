/* strchr (str, ch) -- Return pointer to first occurrence of CH in STR.
   For SPARC v7.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jj@ultra.linux.cz> and
		  David S. Miller <davem@caip.rutgers.edu>.

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
ENTRY(strchr)
	andcc		%o1, 0xff, %o1
	be		12f
	 sll		%o1, 8, %o2
	andcc		%o0, 3, %g0
	or		%o1, %o2, %o2
	sethi		%hi(0x80808080), %o4
	sll		%o2, 16, %o3
	be		13f
	 or		%o3, %o2, %g2

	ldub		[%o0], %g4
	cmp		%g4, %o1
	be		11f
	 add		%o0, 1, %o0
	cmp		%g4, 0
	be		9f
	 andcc		%o0, 3, %g0
	be		4f
	 or		%o4, %lo(0x80808080), %o3
	ldub		[%o0], %g4
	cmp		%g4, %o1
	be		11f
	 add		%o0, 1, %o0
	cmp		%g4, 0
	be		9f
	 andcc		%o0, 3, %g0
	be		5f
	 sethi		%hi(0x01010101), %o5
	ldub		[%o0], %g4
	cmp		%g4, %o1
	be		11f
	 add		%o0, 1, %o0
	cmp		%g4, 0
	be		9f
	 or		%o5, %lo(0x01010101), %o2
	b		6f
	 ld		[%o0], %g4
11:	retl
	 sub		%o0, 1, %o0

13:	or		%o4, %lo(0x80808080), %o3
4:	sethi		%hi(0x01010101), %o5
5:	or		%o5, %lo(0x01010101), %o2
7:	ld		[%o0], %g4
6:	xor		%g4, %g2, %g5
	sub		%g4, %o2, %o4
#ifdef EIGHTBIT_NOT_RARE
	sub		%g5, %o2, %g6
	andn		%o4, %g4, %o4
	andn		%g6, %g5, %g5
#else
	sub		%g5, %o2, %g5
#endif
	or		%g5, %o4, %o4
	andcc		%o4, %o3, %g0
	be		7b
	 add		%o0, 4, %o0

	/* Check every byte. */
8:	srl		%g4, 24, %g5
7:	andcc		%g5, 0xff, %g5
	be		9f
	 cmp		%g5, %o1
	be		4f
	 srl		%g4, 16, %g5
	andcc		%g5, 0xff, %g5
	be		9f
	 cmp		%g5, %o1
	be		3f
	 srl		%g4, 8, %g5
	andcc		%g5, 0xff, %g5
	be		9f
	 cmp		%g5, %o1
	be		2f
	 andcc		%g4, 0xff, %g5
	be		9f
	 cmp		%g5, %o1
	bne,a		6b
	 ld		[%o0], %g4
	retl
	 sub		%o0, 1, %o0
2:	retl
	 sub		%o0, 2, %o0
3:	retl
	 sub		%o0, 3, %o0
4:	retl
	 sub		%o0, 4, %o0
9:	retl
	 clr		%o0

11:	ldub		[%o0], %o5
	cmp		%o5, 0
	be		1f
	 add		%o0, 1, %o0
	andcc		%o0, 3, %g0
	be		4f
	 or		%o4, %lo(0x80808080), %o3
	ldub		[%o0], %o5
	cmp		%o5, 0
	be		1f
	 add		%o0, 1, %o0
	andcc		%o0, 3, %g0
	be		5f
	 sethi		%hi(0x01010101), %o4
	ldub		[%o0], %o5
	cmp		%o5, 0
	be		1f
	 add		%o0, 1, %o0
	b		6f
	 or		%o4, %lo(0x01010101), %o2
1:	retl
	 sub		%o0, 1, %o0

12:	andcc		%o0, 3, %g0
	bne		11b
	 sethi		%hi(0x80808080), %o4
	or		%o4, %lo(0x80808080), %o3
4:	sethi		%hi(0x01010101), %o4
5:	or		%o4, %lo(0x01010101), %o2
6:	ld		[%o0], %o5
7:	sub		%o5, %o2, %o4
#ifdef EIGHTBIT_NOT_RARE
	andn		%o4, %o5, %o4
#endif
	andcc		%o4, %o3, %g0
	be		6b
	 add		%o0, 4, %o0

	/* Check every byte. */
	srl		%o5, 24, %g5
	andcc		%g5, 0xff, %g0
	be		8f
	 add		%o0, -4, %o4
	srl		%o5, 16, %g5
	andcc		%g5, 0xff, %g0
	be		8f
	 add		%o4, 1, %o4
	srl		%o5, 8, %g5
	andcc		%g5, 0xff, %g0
	be		8f
	 add		%o4, 1, %o4
	andcc		%o5, 0xff, %g0
	bne,a		7b
	 ld		[%o0], %o5
	add		%o4, 1, %o4
8:	retl
	 mov		%o4, %o0

13:	ldub		[%o0], %g4
	cmp		%g4, %o1
	add		%o0, 1, %o0
	be,a		1f
	 sub		%o0, 1, %o5
	cmp		%g4, 0
	be		9f
1:	 andcc		%o0, 3, %g0
	be		4f
	 or		%o4, %lo(0x80808080), %o3
	ldub		[%o0], %g4
	cmp		%g4, %o1
	add		%o0, 1, %o0
	be,a		1f
	 sub		%o0, 1, %o5
	cmp		%g4, 0
	be		9f
1:	 andcc		%o0, 3, %g0
	be		5f
	 sethi		%hi(0x01010101), %o4
	ldub		[%o0], %g4
	cmp		%g4, %o1
	add		%o0, 1, %o0
	be,a		1f
	 sub		%o0, 1, %o5
	cmp		%g4, 0
	be		9f
1:	 or		%o4, %lo(0x01010101), %o2
	b		7f
	 ld		[%o0], %g4
END(strchr)

ENTRY(strrchr)
	andcc		%o1, 0xff, %o1
	clr		%o5
	be		12b
	 sll		%o1, 8, %o2
	andcc		%o0, 3, %g0
	or		%o1, %o2, %o2
	sethi		%hi(0x80808080), %o4
	sll		%o2, 16, %o3
	bne		13b
	 or		%o3, %o2, %g2
	or		%o4, %lo(0x80808080), %o3
4:	sethi		%hi(0x01010101), %o4
5:	or		%o4, %lo(0x01010101), %o2
6:	ld		[%o0], %g4
7:	xor		%g4, %g2, %g5
	sub		%g4, %o2, %o4
#ifdef EIGHTBIT_NOT_RARE
	sub		%g5, %o2, %g6
	andn		%o4, %g4, %o4
	andn		%g6, %g5, %g5
#else
	sub		%g5, %o2, %g5
#endif
	or		%g5, %o4, %o4
	andcc		%o4, %o3, %g0
	be		6b
	 add		%o0, 4, %o0

	/* Check every byte. */
3:	srl		%g4, 24, %g5
8:	andcc		%g5, 0xff, %g5
	be		9f
	 cmp		%g5, %o1
	be,a		1f
	 sub		%o0, 4, %o5
1:	srl		%g4, 16, %g5
	andcc		%g5, 0xff, %g5
	be		9f
	 cmp		%g5, %o1
	be,a		1f
	 sub		%o0, 3, %o5
1:	srl		%g4, 8, %g5
	andcc		%g5, 0xff, %g5
	be		9f
	 cmp		%g5, %o1
	be,a		1f
	 sub		%o0, 2, %o5
1:	andcc		%g4, 0xff, %g5
	be		9f
	 cmp		%g5, %o1
	be,a		1f
	 sub		%o0, 1, %o5
1:	b		7b
	 ld		[%o0], %g4
9:	retl
	 mov		%o5, %o0
END(strrchr)

weak_alias (strchr, index)
weak_alias (strrchr, rindex)
libc_hidden_builtin_def (strchr)
libc_hidden_builtin_def (strrchr)
