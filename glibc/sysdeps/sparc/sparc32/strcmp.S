/* Compare two strings for differences.
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

ENTRY(strcmp)
	andcc		%o0, 3, %g0
	be		13f
	 sethi		%hi(0x80808080), %g1

	ldub		[%o0], %o4
	add		%o0, 1, %o0
	ldub		[%o1], %o5
	cmp		%o4, 0
	add		%o1, 1, %o1
	be		2f
	 subcc		%o4, %o5, %o4
	bne		2f
	 andcc		%o0, 3, %g0
	be		4f
	 or		%g1, %lo(0x80808080), %o3
	ldub		[%o0], %o4
	add		%o0, 1, %o0
	ldub		[%o1], %o5
	cmp		%o4, 0
	add		%o1, 1, %o1
	be		2f
	 subcc		%o4, %o5, %o4
	bne		2f
	 andcc		%o0, 3, %g0
	be		5f
	 sethi		%hi(0x01010101), %g1
	ldub		[%o0], %o4
	add		%o0, 1, %o0
	ldub		[%o1], %o5
	cmp		%o4, 0
	add		%o1, 1, %o1
	be		2f
	 subcc		%o4, %o5, %o4
	bne		2f
	 andcc		%o1, 3, %g2
	bne		12f
	 or		%g1, %lo(0x01010101), %o2
	b		1f
	 ld		[%o0], %o4
2:	retl
	 mov		%o4, %o0

13:	or		%g1, %lo(0x80808080), %o3
4:	sethi		%hi(0x01010101), %g1
5:	andcc		%o1, 3, %g2
	bne		12f
	 or		%g1, %lo(0x01010101), %o2

0:	ld		[%o0], %o4
1:	ld		[%o1], %o5
	sub		%o4, %o2, %g1
	add		%o0, 4, %o0
	cmp		%o4, %o5
#ifdef EIGHTBIT_NOT_RARE
	andn		%g1, %o4, %g1
#endif
	bne		11f
	 andcc		%g1, %o3, %g0
	be		0b
	 add		%o1, 4, %o1

	srl		%o4, 24, %g4
	andcc		%g4, 0xff, %g0
	be		2f
	 srl		%o4, 16, %g4
	andcc		%g4, 0xff, %g0
	be		2f
	 srl		%o4, 8, %g4
	andcc		%g4, 0xff, %g0
	be		2f
	 andcc		%o4, 0xff, %g0
	bne,a		1b
	 ld		[%o0], %o4
2:	retl
	 clr		%o0

11:	srl		%o4, 24, %g4
	srl		%o5, 24, %g5
	andcc		%g4, 0xff, %g0
	be		3f
	 subcc		%g4, %g5, %g4
	bne		3f
	 srl		%o5, 16, %g5
	srl		%o4, 16, %g4
	andcc		%g4, 0xff, %g0
	be		3f
	 subcc		%g4, %g5, %g4
	bne		3f
	 srl		%o5, 8, %g5
	srl		%o4, 8, %g4
	andcc		%g4, 0xff, %g0
	be		3f
	 subcc		%g4, %g5, %g4
	bne		3f
	 subcc		%o4, %o5, %o4
	retl
	 mov		%o4, %o0
3:	retl
	 mov		%g4, %o0

12:	save		%sp, -64, %sp
	ld		[%i0], %i4
	sll		%g2, 3, %g3
	andn		%i1, 3, %i1
	mov		32, %l1
	ld		[%i1], %l2
	mov		-1, %g6
	add		%i1, 4, %i1
	sub		%l1, %g3, %l1
	sll		%g6, %g3, %g6

1:	sll		%l2, %g3, %g5
	and		%i4, %g6, %l3
	sub		%i4, %i2, %g1
#ifdef EIGHTBIT_NOT_RARE
	andn		%g1, %i4, %g1
#endif
	andcc		%g1, %i3, %g1
	bne		3f
	 cmp		%g5, %l3
	bne		2f
	 add		%i0, 4, %i0
	ld		[%i1], %l2
	add		%i1, 4, %i1
	srl		%l2, %l1, %l4
	or		%l4, %g5, %l4
	cmp		%l4, %i4
	be,a		1b
	 ld		[%i0], %i4
	restore		%l4, %g0, %o3
	retl
	 sub		%o4, %o3, %o0

2:	sll		%l2, %g3, %i2
	srl		%i4, %g3, %i3
	srl		%i2, %g3, %i2
	restore
	retl
	 sub		%o3, %o2, %o0

3:	srl		%i4, 24, %g4
	srl		%g5, 24, %l6
	andcc		%g4, 0xff, %g0
	be		4f
	 subcc		%g4, %l6, %g4
	bne		4f
	 cmp		%g2, 3
	be		6f
	 srl		%i4, 16, %g4
	srl		%g5, 16, %l6
	andcc		%g4, 0xff, %g0
	be		4f
	 subcc		%g4, %l6, %g4
	bne		4f
	 cmp		%g2, 2
	be		5f
	 srl		%i4, 8, %g4
	srl		%g5, 8, %l6
	andcc		%g4, 0xff, %g0
	be		4f
	 subcc		%g4, %l6, %g4
	bne		4f
	 add		%i0, 4, %i0
	ld		[%i1], %l2
	add		%i1, 4, %i1
	srl		%l2, 24, %g5
	andcc		%i4, 0xff, %g4
	be		4f
	 subcc		%g4, %g5, %g4
	be,a		1b
	 ld		[%i0], %i4
4:	jmpl		%i7 + 8, %g0
	 restore	%g4, %g0, %o0

5:	ld		[%i1], %l2
	add		%i1, 4, %i1
	add		%i0, 4, %i0
	srl		%l2, 24, %l6
	andcc		%g4, 0xff, %g4
	be		4b
	 subcc		%g4, %l6, %g4
	bne		4b
	 srl		%l2, 16, %l6
	andcc		%i4, 0xff, %g4
	and		%l6, 0xff, %l6
	be		4b
	 subcc		%g4, %l6, %g4
	be,a		1b
	 ld		[%i0], %i4
	jmpl		%i7 + 8, %g0
	 restore	%g4, %g0, %o0

6:	ld		[%i1], %l2
	add		%i1, 4, %i1
	add		%i0, 4, %i0
	srl		%l2, 24, %l6
	andcc		%g4, 0xff, %g4
	be		4b
	 subcc		%g4, %l6, %g4
	bne		4b
	 srl		%l2, 16, %l6
	srl		%i4, 8, %g4
	and		%l6, 0xff, %l6
	andcc		%g4, 0xff, %g4
	be		4b
	 subcc		%g4, %l6, %g4
	bne		4b
	 srl		%l2, 8, %l6
	andcc		%i4, 0xff, %g4
	and		%l6, 0xff, %l6
	be		4b
	 subcc		%g4, %l6, %g4
	be,a		1b
	 ld		[%i0], %i4
	jmpl		%i7 + 8, %g0
	 restore	%g4, %g0, %o0
END(strcmp)
libc_hidden_builtin_def (strcmp)
