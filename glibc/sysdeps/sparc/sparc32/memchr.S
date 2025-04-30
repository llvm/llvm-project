/* memchr (str, ch, n) -- Return pointer to first occurrence of CH in STR less
   than N.
   For SPARC v7.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jj@ultra.linux.cz> and
		  David S. Miller <davem@caip.rutgers.edu>.
   This version is developed using the same algorithm as the fast C
   version which carries the following introduction:
   Based on strlen implementation by Torbjorn Granlund (tege@sics.se),
   with help from Dan Sahlin (dan@sics.se) and
   commentary by Jim Blandy (jimb@ai.mit.edu);
   adaptation to memchr suggested by Dick Karpinski (dick@cca.ucsf.edu),
   and implemented by Roland McGrath (roland@ai.mit.edu).

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

	.text
	.align		4
ENTRY(__memchr)
	andcc		%o1, 0xff, %o1
	sll		%o1, 8, %g6
	andcc		%o0, 3, %g0
	or		%o1, %g6, %g6
	sll		%g6, 16, %o3
	be		10f
	 or		%o3, %g6, %g2
	cmp		%o2, 0
	be		9f
	 sethi		%hi(0x80808080), %o4
	ldub		[%o0], %g4
	cmp		%g4, %o1
	be		1f
	 add		%o0, 1, %o0
	subcc		%o2, 1, %o2
	be		9f
	 andcc		%o0, 3, %g0
	be		4f
	 or		%o4, %lo(0x80808080), %o3
	ldub		[%o0], %g4
	cmp		%g4, %o1
	be		1f
	 add		%o0, 1, %o0
	subcc		%o2, 1, %o2
	be		9f
	 andcc		%o0, 3, %g0
	be		5f
	 sethi		%hi(0x01010101), %o5
	ldub		[%o0], %g4
	cmp		%g4, %o1
	be		1f
	 add		%o0, 1, %o0
	subcc		%o2, 1, %o2
	bne,a		7f
	 and		%o2, 3, %g1
	retl
	 clr		%o0
1:	retl
	 sub		%o0, 1, %o0
10:	sethi		%hi(0x80808080), %o4
	or		%o4, %lo(0x80808080), %o3
4:	sethi		%hi(0x01010101), %o5
5:	and		%o2, 3, %g1
7:	andcc		%o2, 0xfffffffc, %o2
	be		0f
	 or		%o5, %lo(0x01010101), %g6
	ld		[%o0], %g4
6:	xor		%g4, %g2, %g5
	add		%o0, 4, %o0
	sub		%g5, %g6, %g5
	andcc		%g5, %o3, %g0
	bne		8f
	 subcc		%o2, 4, %o2
	bne,a		6b
	 ld		[%o0], %g4
0:	cmp		%g1, 0
1:	be		9f
	 add		%o0, 4, %o0
	ldub		[%o0 - 4], %g4
	cmp		%g4, %o1
	be		4f
	 cmp		%g1, 1
	be		9f
	 ldub		[%o0 - 3], %g4
	cmp		%g4, %o1
	be		3f
	 cmp		%g1, 2
	be		9f
	 ldub		[%o0 - 2], %g4
	cmp		%g4, %o1
	be		2f
	 nop
9:	retl
	 clr		%o0

	/* Check every byte. */
8:	srl		%g4, 24, %g5
	and		%g5, 0xff, %g5
	cmp		%g5, %o1
	be		4f
	 srl		%g4, 16, %g5
	and		%g5, 0xff, %g5
	cmp		%g5, %o1
	be		3f
	 srl		%g4, 8, %g5
	and		%g5, 0xff, %g5
	cmp		%g5, %o1
	be		2f
	 and		%g4, 0xff, %g5
	cmp		%g5, %o1
	be		1f
	 cmp		%o2, 0
	bne,a		6b
	 ld		[%o0], %g4
	b		1b
	 cmp		%g1, 0
1:	retl
	 sub		%o0, 1, %o0
2:	retl
	 sub		%o0, 2, %o0
3:	retl
	 sub		%o0, 3, %o0
4:	retl
	 sub		%o0, 4, %o0
END(__memchr)

weak_alias (__memchr, memchr)
libc_hidden_builtin_def (memchr)
