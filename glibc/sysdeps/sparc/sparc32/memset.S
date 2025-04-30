/* Set a block of memory to some byte value.
   For SPARC v7.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David S. Miller <davem@caip.rutgers.edu> and
                  Jakub Jelinek <jj@ultra.linux.cz>.

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

	/* Store 64 bytes at (BASE + OFFSET) using value SOURCE. */
#define ZERO_BIG_BLOCK(base, offset, source)    	\
	std		source, [base + offset + 0x00]; \
	std		source, [base + offset + 0x08]; \
	std		source, [base + offset + 0x10]; \
	std		source, [base + offset + 0x18]; \
	std		source, [base + offset + 0x20]; \
	std		source, [base + offset + 0x28]; \
	std		source, [base + offset + 0x30]; \
	std		source, [base + offset + 0x38];

#define ZERO_LAST_BLOCKS(base, offset, source)		\
	std		source, [base - offset - 0x38]; \
	std		source, [base - offset - 0x30]; \
	std		source, [base - offset - 0x28]; \
	std		source, [base - offset - 0x20]; \
	std		source, [base - offset - 0x18]; \
	std		source, [base - offset - 0x10]; \
	std		source, [base - offset - 0x08]; \
	std		source, [base - offset - 0x00];

	.text
	.align 4
ENTRY(__bzero)
	b		1f
	 mov		%g0, %g3

3:	cmp		%o2, 3
	be		2f
	 stb		%g3, [%o0]

	cmp		%o2, 2
	be		2f
	 stb		%g3, [%o0 + 0x01]

	stb		%g3, [%o0 + 0x02]
2:	sub		%o2, 4, %o2
	add		%o1, %o2, %o1
	b		4f
	 sub		%o0, %o2, %o0
END(__bzero)

ENTRY(memset)
	and		%o1, 0xff, %g3
	sll		%g3, 8, %g2
	or		%g3, %g2, %g3
	sll		%g3, 16, %g2
	or		%g3, %g2, %g3
	orcc		%o2, %g0, %o1
1:	cmp		%o1, 7
	bleu		7f
	 mov		%o0, %g1

	andcc		%o0, 3, %o2
	bne		3b
4:	 andcc		%o0, 4, %g0

	be		2f
	 mov		%g3, %g2

	st		%g3, [%o0]
	sub		%o1, 4, %o1
	add		%o0, 4, %o0
2:	andcc		%o1, 0xffffff80, %o3
	be		9f
	 andcc		%o1, 0x78, %o2
4:	ZERO_BIG_BLOCK	(%o0, 0x00, %g2)
	subcc		%o3, 128, %o3
	ZERO_BIG_BLOCK	(%o0, 0x40, %g2)
	bne		4b
	 add		%o0, 128, %o0

	orcc		%o2, %g0, %g0
9:	be		6f
	 andcc		%o1, 7, %o1

	mov		%o7, %g4
101:	call		100f
	 srl		%o2, 1, %o3
	mov		%g4, %o7
	jmpl		%o4 + (20f + 64 - 101b), %g0
	 add		%o0, %o2, %o0

100:	retl
	 sub		%o7, %o3, %o4

20:	ZERO_LAST_BLOCKS(%o0, 0x48, %g2)
	ZERO_LAST_BLOCKS(%o0, 0x08, %g2)

6:	be		8f
	 andcc		%o1, 4, %g0
	be		1f
	 andcc		%o1, 2, %g0
	st		%g3, [%o0]
	add		%o0, 4, %o0
1:	be		1f
	 andcc		%o1, 1, %g0
	sth		%g3, [%o0]
	add		%o0, 2, %o0
1:	bne,a		8f
	 stb		%g3, [%o0]
8:	retl
	 mov		%g1, %o0
7:	orcc		%o1, 0, %g0
	be		0f
	 subcc		%o1, 1, %o1
	stb		%g3, [%o0]
	be		0f
	 subcc		%o1, 1, %o1
	stb		%g3, [%o0 + 1]
	be		0f
	 subcc		%o1, 1, %o1
	stb		%g3, [%o0 + 2]
	be		0f
	 subcc		%o1, 1, %o1
	stb		%g3, [%o0 + 3]
	be		0f
	 subcc		%o1, 1, %o1
	stb		%g3, [%o0 + 4]
	be		0f
	 subcc		%o1, 1, %o1
	stb		%g3, [%o0 + 5]
	be		0f
	 subcc		%o1, 1, %o1
	stb		%g3, [%o0 + 6]
0:	retl
	 nop
END(memset)
libc_hidden_builtin_def (memset)

weak_alias (__bzero, bzero)
