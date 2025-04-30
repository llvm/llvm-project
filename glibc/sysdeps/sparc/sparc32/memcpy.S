/* Copy SIZE bytes from SRC to DEST.
   For SPARC v7.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David S. Miller <davem@caip.rutgers.edu>,
		  Eddie C. Dost <ecd@skynet.be> and
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

/* Both these macros have to start with exactly the same insn */
#define MOVE_BIGCHUNK(src, dst, offset, t0, t1, t2, t3, t4, t5, t6, t7) 				\
	ldd	[%src + offset + 0x00], %t0; 								\
	ldd	[%src + offset + 0x08], %t2; 								\
	ldd	[%src + offset + 0x10], %t4; 								\
	ldd	[%src + offset + 0x18], %t6; 								\
	st	%t0, [%dst + offset + 0x00]; 								\
	st	%t1, [%dst + offset + 0x04]; 								\
	st	%t2, [%dst + offset + 0x08]; 								\
	st	%t3, [%dst + offset + 0x0c]; 								\
	st	%t4, [%dst + offset + 0x10]; 								\
	st	%t5, [%dst + offset + 0x14]; 								\
	st	%t6, [%dst + offset + 0x18]; 								\
	st	%t7, [%dst + offset + 0x1c];

#define MOVE_BIGALIGNCHUNK(src, dst, offset, t0, t1, t2, t3, t4, t5, t6, t7) 				\
	ldd	[%src + offset + 0x00], %t0; 								\
	ldd	[%src + offset + 0x08], %t2; 								\
	ldd	[%src + offset + 0x10], %t4; 								\
	ldd	[%src + offset + 0x18], %t6; 								\
	std	%t0, [%dst + offset + 0x00]; 								\
	std	%t2, [%dst + offset + 0x08]; 								\
	std	%t4, [%dst + offset + 0x10]; 								\
	std	%t6, [%dst + offset + 0x18];

#define MOVE_LASTCHUNK(src, dst, offset, t0, t1, t2, t3) 						\
	ldd	[%src - offset - 0x10], %t0; 								\
	ldd	[%src - offset - 0x08], %t2; 								\
	st	%t0, [%dst - offset - 0x10]; 								\
	st	%t1, [%dst - offset - 0x0c]; 								\
	st	%t2, [%dst - offset - 0x08]; 								\
	st	%t3, [%dst - offset - 0x04];

#define MOVE_LASTALIGNCHUNK(src, dst, offset, t0, t1, t2, t3) 						\
	ldd	[%src - offset - 0x10], %t0; 								\
	ldd	[%src - offset - 0x08], %t2; 								\
	std	%t0, [%dst - offset - 0x10]; 								\
	std	%t2, [%dst - offset - 0x08];

#define MOVE_SHORTCHUNK(src, dst, offset, t0, t1) 							\
	ldub	[%src - offset - 0x02], %t0; 								\
	ldub	[%src - offset - 0x01], %t1; 								\
	stb	%t0, [%dst - offset - 0x02]; 								\
	stb	%t1, [%dst - offset - 0x01];

#define SMOVE_CHUNK(src, dst, offset, t0, t1, t2, t3, t4, t5, t6, prev, shil, shir, offset2) 		\
	ldd	[%src + offset + 0x00], %t0; 								\
	ldd	[%src + offset + 0x08], %t2; 								\
	srl	%t0, shir, %t5; 									\
	srl	%t1, shir, %t6; 									\
	sll	%t0, shil, %t0; 									\
	or	%t5, %prev, %t5; 									\
	sll	%t1, shil, %prev; 									\
	or	%t6, %t0, %t0; 										\
	srl	%t2, shir, %t1; 									\
	srl	%t3, shir, %t6; 									\
	sll	%t2, shil, %t2; 									\
	or	%t1, %prev, %t1; 									\
	std	%t4, [%dst + offset + offset2 - 0x04]; 							\
	std	%t0, [%dst + offset + offset2 + 0x04];							\
	sll	%t3, shil, %prev; 									\
	or	%t6, %t2, %t4;

#define SMOVE_ALIGNCHUNK(src, dst, offset, t0, t1, t2, t3, t4, t5, t6, prev, shil, shir, offset2) 	\
	ldd	[%src + offset + 0x00], %t0; 								\
	ldd	[%src + offset + 0x08], %t2; 								\
	srl	%t0, shir, %t4; 									\
	srl	%t1, shir, %t5; 									\
	sll	%t0, shil, %t6; 									\
	or	%t4, %prev, %t0; 									\
	sll	%t1, shil, %prev; 									\
	or	%t5, %t6, %t1; 										\
	srl	%t2, shir, %t4; 									\
	srl	%t3, shir, %t5; 									\
	sll	%t2, shil, %t6; 									\
	or	%t4, %prev, %t2; 									\
	sll	%t3, shil, %prev; 									\
	or	%t5, %t6, %t3;										\
	std	%t0, [%dst + offset + offset2 + 0x00]; 							\
	std	%t2, [%dst + offset + offset2 + 0x08];

	.text
ENTRY(__mempcpy)
	add		%o0, %o2, %g1
	ba		101f
	 st		%g1, [%sp + 64]
END(__mempcpy)

	.align	4
ENTRY(memcpy)		/* %o0=dst %o1=src %o2=len */
	st		%o0, [%sp + 64]
101:
	sub		%o0, %o1, %o4
9:	andcc		%o4, 3, %o5
0:	bne		86f
	 cmp		%o2, 15

	bleu		90f
	 andcc		%o1, 3, %g0

	be		78f
	 andcc		%o1, 4, %g0

	andcc		%o1, 1, %g0
	be		4f
	 andcc		%o1, 2, %g0

	ldub		[%o1], %g2
	add		%o1, 1, %o1
	stb		%g2, [%o0]
	sub		%o2, 1, %o2
	bne		77f
	 add		%o0, 1, %o0
4:	lduh		[%o1], %g2
	add		%o1, 2, %o1
	sth		%g2, [%o0]
	sub		%o2, 2, %o2
	add		%o0, 2, %o0

77:	andcc		%o1, 4, %g0
78:	be		2f
	 mov		%o2, %g1

	ld		[%o1], %o4
	sub		%g1, 4, %g1
	st		%o4, [%o0]
	add		%o1, 4, %o1
	add		%o0, 4, %o0
2:	andcc		%g1, 0xffffff80, %g6
	be		3f
	 andcc		%o0, 4, %g0

	be		82f + 4
5:	MOVE_BIGCHUNK(o1, o0, 0x00, o2, o3, o4, o5, g2, g3, g4, g5)
	MOVE_BIGCHUNK(o1, o0, 0x20, o2, o3, o4, o5, g2, g3, g4, g5)
	MOVE_BIGCHUNK(o1, o0, 0x40, o2, o3, o4, o5, g2, g3, g4, g5)
	MOVE_BIGCHUNK(o1, o0, 0x60, o2, o3, o4, o5, g2, g3, g4, g5)
	subcc		%g6, 128, %g6
	add		%o1, 128, %o1
	bne		5b
	 add		%o0, 128, %o0
3:	andcc		%g1, 0x70, %g6
	be		80f
	 andcc		%g1, 8, %g0

	srl		%g6, 1, %o4
	mov		%o7, %g2
	add		%g6, %o4, %o4
	add		%o1, %g6, %o1
104:	call		100f
	 add		%o0, %g6, %o0
	jmpl		%o5 + (80f - 104b), %g0
	 mov		%g2, %o7

79:	MOVE_LASTCHUNK(o1, o0, 0x60, g2, g3, g4, g5)
	MOVE_LASTCHUNK(o1, o0, 0x50, g2, g3, g4, g5)
	MOVE_LASTCHUNK(o1, o0, 0x40, g2, g3, g4, g5)
	MOVE_LASTCHUNK(o1, o0, 0x30, g2, g3, g4, g5)
	MOVE_LASTCHUNK(o1, o0, 0x20, g2, g3, g4, g5)
	MOVE_LASTCHUNK(o1, o0, 0x10, g2, g3, g4, g5)
	MOVE_LASTCHUNK(o1, o0, 0x00, g2, g3, g4, g5)

80:	be		81f
	 andcc		%g1, 4, %g0

	ldd		[%o1], %g2
	add		%o0, 8, %o0
	st		%g2, [%o0 - 0x08]
	add		%o1, 8, %o1
	st		%g3, [%o0 - 0x04]

81:	be		1f
	 andcc		%g1, 2, %g0

	ld		[%o1], %g2
	add		%o1, 4, %o1
	st		%g2, [%o0]
	add		%o0, 4, %o0
1:	be		1f
	 andcc		%g1, 1, %g0

	lduh		[%o1], %g2
	add		%o1, 2, %o1
	sth		%g2, [%o0]
	add		%o0, 2, %o0
1:	be		1f
	 nop

	ldub		[%o1], %g2
	stb		%g2, [%o0]
1:	retl
	 ld		[%sp + 64], %o0

82:	/* ldd_std */
	MOVE_BIGALIGNCHUNK(o1, o0, 0x00, o2, o3, o4, o5, g2, g3, g4, g5)
	MOVE_BIGALIGNCHUNK(o1, o0, 0x20, o2, o3, o4, o5, g2, g3, g4, g5)
	MOVE_BIGALIGNCHUNK(o1, o0, 0x40, o2, o3, o4, o5, g2, g3, g4, g5)
	MOVE_BIGALIGNCHUNK(o1, o0, 0x60, o2, o3, o4, o5, g2, g3, g4, g5)
	subcc		%g6, 128, %g6
	add		%o1, 128, %o1
	bne		82b
	 add		%o0, 128, %o0

	andcc		%g1, 0x70, %g6
	be		84f
	 andcc		%g1, 8, %g0

	mov		%o7, %g2
111:	call		110f
	 add		%o1, %g6, %o1
	mov		%g2, %o7
	jmpl		%o5 + (84f - 111b), %g0
	 add		%o0, %g6, %o0

83:	MOVE_LASTALIGNCHUNK(o1, o0, 0x60, g2, g3, g4, g5)
	MOVE_LASTALIGNCHUNK(o1, o0, 0x50, g2, g3, g4, g5)
	MOVE_LASTALIGNCHUNK(o1, o0, 0x40, g2, g3, g4, g5)
	MOVE_LASTALIGNCHUNK(o1, o0, 0x30, g2, g3, g4, g5)
	MOVE_LASTALIGNCHUNK(o1, o0, 0x20, g2, g3, g4, g5)
	MOVE_LASTALIGNCHUNK(o1, o0, 0x10, g2, g3, g4, g5)
	MOVE_LASTALIGNCHUNK(o1, o0, 0x00, g2, g3, g4, g5)

84:	be		85f
	 andcc		%g1, 4, %g0

	ldd		[%o1], %g2
	add		%o0, 8, %o0
	std		%g2, [%o0 - 0x08]
	add		%o1, 8, %o1
85:	be		1f
	 andcc		%g1, 2, %g0

	ld		[%o1], %g2
	add		%o1, 4, %o1
	st		%g2, [%o0]
	add		%o0, 4, %o0
1:	be		1f
	 andcc		%g1, 1, %g0

	lduh		[%o1], %g2
	add		%o1, 2, %o1
	sth		%g2, [%o0]
	add		%o0, 2, %o0
1:	be		1f
	 nop

	ldub		[%o1], %g2
	stb		%g2, [%o0]
1:	retl
	 ld		[%sp + 64], %o0

86:	cmp		%o2, 6
	bleu		88f

	 cmp		%o2, 256
	bcc		87f

	 andcc		%o0, 3, %g0
	be		61f
	 andcc		%o0, 1, %g0
	be		60f
	 andcc		%o0, 2, %g0

	ldub		[%o1], %g5
	add		%o1, 1, %o1
	stb		%g5, [%o0]
	sub		%o2, 1, %o2
	bne		61f
	 add		%o0, 1, %o0
60:	ldub		[%o1], %g3
	add		%o1, 2, %o1
	stb		%g3, [%o0]
	sub		%o2, 2, %o2
	ldub		[%o1 - 1], %g3
	add		%o0, 2, %o0
	stb		%g3, [%o0 - 1]
61:	and		%o1, 3, %g2
	and		%o2, 0xc, %g3
	and		%o1, -4, %o1
	cmp		%g3, 4
	sll		%g2, 3, %g4
	mov		32, %g2
	be		4f
	 sub		%g2, %g4, %g6

	blu		3f
	 cmp		%g3, 0x8

	be		2f
	 srl		%o2, 2, %g3

	ld		[%o1], %o3
	add		%o0, -8, %o0
	ld		[%o1 + 4], %o4
	b		8f
	 add		%g3, 1, %g3
2:	ld		[%o1], %o4
	add		%o0, -12, %o0
	ld		[%o1 + 4], %o5
	add		%g3, 2, %g3
	b		9f
	 add		%o1, -4, %o1
3:	ld		[%o1], %g1
	add		%o0, -4, %o0
	ld		[%o1 + 4], %o3
	srl		%o2, 2, %g3
	b		7f
	 add		%o1, 4, %o1
4:	ld		[%o1], %o5
	cmp		%o2, 7
	ld		[%o1 + 4], %g1
	srl		%o2, 2, %g3
	bleu		10f
	 add		%o1, 8, %o1

	ld		[%o1], %o3
	add		%g3, -1, %g3
5:	sll		%o5, %g4, %g2
	srl		%g1, %g6, %g5
	or		%g2, %g5, %g2
	st		%g2, [%o0]
7:	ld		[%o1 + 4], %o4
	sll		%g1, %g4, %g2
	srl		%o3, %g6, %g5
	or		%g2, %g5, %g2
	st		%g2, [%o0 + 4]
8:	ld		[%o1 + 8], %o5
	sll		%o3, %g4, %g2
	srl		%o4, %g6, %g5
	or		%g2, %g5, %g2
	st		%g2, [%o0 + 8]
9:	ld		[%o1 + 12], %g1
	sll		%o4, %g4, %g2
	srl		%o5, %g6, %g5
	addcc		%g3, -4, %g3
	or		%g2, %g5, %g2
	add		%o1, 16, %o1
	st		%g2, [%o0 + 12]
	add		%o0, 16, %o0
	bne,a		5b
	 ld		[%o1], %o3
10:	sll		%o5, %g4, %g2
	srl		%g1, %g6, %g5
	srl		%g6, 3, %g3
	or		%g2, %g5, %g2
	sub		%o1, %g3, %o1
	andcc		%o2, 2, %g0
	st		%g2, [%o0]
	be		1f
	 andcc		%o2, 1, %g0

	ldub		[%o1], %g2
	add		%o1, 2, %o1
	stb		%g2, [%o0 + 4]
	add		%o0, 2, %o0
	ldub		[%o1 - 1], %g2
	stb		%g2, [%o0 + 3]
1:	be		1f
	 nop
	ldub		[%o1], %g2
	stb		%g2, [%o0 + 4]
1:	retl
	 ld		[%sp + 64], %o0

87:	andcc		%o1, 3, %g0
	be		3f
	 andcc		%o1, 1, %g0

	be		4f
	 andcc		%o1, 2, %g0

	ldub		[%o1], %g2
	add		%o1, 1, %o1
	stb		%g2, [%o0]
	sub		%o2, 1, %o2
	bne		3f
	 add		%o0, 1, %o0
4:	lduh		[%o1], %g2
	add		%o1, 2, %o1
	srl		%g2, 8, %g3
	sub		%o2, 2, %o2
	stb		%g3, [%o0]
	add		%o0, 2, %o0
	stb		%g2, [%o0 - 1]
3:	 andcc		%o1, 4, %g0

	bne		2f
	 cmp		%o5, 1

	ld		[%o1], %o4
	srl		%o4, 24, %g2
	stb		%g2, [%o0]
	srl		%o4, 16, %g3
	stb		%g3, [%o0 + 1]
	srl		%o4, 8, %g2
	stb		%g2, [%o0 + 2]
	sub		%o2, 4, %o2
	stb		%o4, [%o0 + 3]
	add		%o1, 4, %o1
	add		%o0, 4, %o0
2:	be		33f
	 cmp		%o5, 2
	be		32f
	 sub		%o2, 4, %o2
31:	ld		[%o1], %g2
	add		%o1, 4, %o1
	srl		%g2, 24, %g3
	and		%o0, 7, %g5
	stb		%g3, [%o0]
	cmp		%g5, 7
	sll		%g2, 8, %g1
	add		%o0, 4, %o0
	be		41f
	 and		%o2, 0xffffffc0, %o3
	ld		[%o0 - 7], %o4
4:	SMOVE_CHUNK(o1, o0, 0x00, g2, g3, g4, g5, o4, o5, g6, g1, 8, 24, -3)
	SMOVE_CHUNK(o1, o0, 0x10, g2, g3, g4, g5, o4, o5, g6, g1, 8, 24, -3)
	SMOVE_CHUNK(o1, o0, 0x20, g2, g3, g4, g5, o4, o5, g6, g1, 8, 24, -3)
	SMOVE_CHUNK(o1, o0, 0x30, g2, g3, g4, g5, o4, o5, g6, g1, 8, 24, -3)
	subcc		%o3, 64, %o3
	add		%o1, 64, %o1
	bne		4b
	 add		%o0, 64, %o0

	andcc		%o2, 0x30, %o3
	be,a		1f
	 srl		%g1, 16, %g2
4:	SMOVE_CHUNK(o1, o0, 0x00, g2, g3, g4, g5, o4, o5, g6, g1, 8, 24, -3)
	subcc		%o3, 16, %o3
	add		%o1, 16, %o1
	bne		4b
	 add		%o0, 16, %o0

	srl		%g1, 16, %g2
1:	st		%o4, [%o0 - 7]
	sth		%g2, [%o0 - 3]
	srl		%g1, 8, %g4
	b		88f
	 stb		%g4, [%o0 - 1]
32:	ld		[%o1], %g2
	add		%o1, 4, %o1
	srl		%g2, 16, %g3
	and		%o0, 7, %g5
	sth		%g3, [%o0]
	cmp		%g5, 6
	sll		%g2, 16, %g1
	add		%o0, 4, %o0
	be		42f
	 and		%o2, 0xffffffc0, %o3
	ld		[%o0 - 6], %o4
4:	SMOVE_CHUNK(o1, o0, 0x00, g2, g3, g4, g5, o4, o5, g6, g1, 16, 16, -2)
	SMOVE_CHUNK(o1, o0, 0x10, g2, g3, g4, g5, o4, o5, g6, g1, 16, 16, -2)
	SMOVE_CHUNK(o1, o0, 0x20, g2, g3, g4, g5, o4, o5, g6, g1, 16, 16, -2)
	SMOVE_CHUNK(o1, o0, 0x30, g2, g3, g4, g5, o4, o5, g6, g1, 16, 16, -2)
	subcc		%o3, 64, %o3
	add		%o1, 64, %o1
	bne		4b
	 add		%o0, 64, %o0

	andcc		%o2, 0x30, %o3
	be,a		1f
	 srl		%g1, 16, %g2
4:	SMOVE_CHUNK(o1, o0, 0x00, g2, g3, g4, g5, o4, o5, g6, g1, 16, 16, -2)
	subcc		%o3, 16, %o3
	add		%o1, 16, %o1
	bne		4b
	 add		%o0, 16, %o0

	srl		%g1, 16, %g2
1:	st		%o4, [%o0 - 6]
	b		88f
	 sth		%g2, [%o0 - 2]
33:	ld		[%o1], %g2
	sub		%o2, 4, %o2
	srl		%g2, 24, %g3
	and		%o0, 7, %g5
	stb		%g3, [%o0]
	cmp		%g5, 5
	srl		%g2, 8, %g4
	sll		%g2, 24, %g1
	sth		%g4, [%o0 + 1]
	add		%o1, 4, %o1
	be		43f
	 and		%o2, 0xffffffc0, %o3

	ld		[%o0 - 1], %o4
	add		%o0, 4, %o0
4:	SMOVE_CHUNK(o1, o0, 0x00, g2, g3, g4, g5, o4, o5, g6, g1, 24, 8, -1)
	SMOVE_CHUNK(o1, o0, 0x10, g2, g3, g4, g5, o4, o5, g6, g1, 24, 8, -1)
	SMOVE_CHUNK(o1, o0, 0x20, g2, g3, g4, g5, o4, o5, g6, g1, 24, 8, -1)
	SMOVE_CHUNK(o1, o0, 0x30, g2, g3, g4, g5, o4, o5, g6, g1, 24, 8, -1)
	subcc		%o3, 64, %o3
	add		%o1, 64, %o1
	bne		4b
	 add		%o0, 64, %o0

	andcc		%o2, 0x30, %o3
	be,a		1f
	 srl		%g1, 24, %g2
4:	SMOVE_CHUNK(o1, o0, 0x00, g2, g3, g4, g5, o4, o5, g6, g1, 24, 8, -1)
	subcc		%o3, 16, %o3
	add		%o1, 16, %o1
	bne		4b
	 add		%o0, 16, %o0

	srl		%g1, 24, %g2
1:	st		%o4, [%o0 - 5]
	b		88f
	 stb		%g2, [%o0 - 1]
41:	SMOVE_ALIGNCHUNK(o1, o0, 0x00, g2, g3, g4, g5, o4, o5, g6, g1, 8, 24, -3)
	SMOVE_ALIGNCHUNK(o1, o0, 0x10, g2, g3, g4, g5, o4, o5, g6, g1, 8, 24, -3)
	SMOVE_ALIGNCHUNK(o1, o0, 0x20, g2, g3, g4, g5, o4, o5, g6, g1, 8, 24, -3)
	SMOVE_ALIGNCHUNK(o1, o0, 0x30, g2, g3, g4, g5, o4, o5, g6, g1, 8, 24, -3)
	subcc		%o3, 64, %o3
	add		%o1, 64, %o1
	bne		41b
	 add		%o0, 64, %o0

	andcc		%o2, 0x30, %o3
	be,a		1f
	 srl		%g1, 16, %g2
4:	SMOVE_ALIGNCHUNK(o1, o0, 0x00, g2, g3, g4, g5, o4, o5, g6, g1, 8, 24, -3)
	subcc		%o3, 16, %o3
	add		%o1, 16, %o1
	bne		4b
	 add		%o0, 16, %o0

	srl		%g1, 16, %g2
1:	sth		%g2, [%o0 - 3]
	srl		%g1, 8, %g4
	b		88f
	 stb		%g4, [%o0 - 1]
43:	SMOVE_ALIGNCHUNK(o1, o0, 0x00, g2, g3, g4, g5, o4, o5, g6, g1, 24, 8, 3)
	SMOVE_ALIGNCHUNK(o1, o0, 0x10, g2, g3, g4, g5, o4, o5, g6, g1, 24, 8, 3)
	SMOVE_ALIGNCHUNK(o1, o0, 0x20, g2, g3, g4, g5, o4, o5, g6, g1, 24, 8, 3)
	SMOVE_ALIGNCHUNK(o1, o0, 0x30, g2, g3, g4, g5, o4, o5, g6, g1, 24, 8, 3)
	subcc		%o3, 64, %o3
	add		%o1, 64, %o1
	bne		43b
	 add		%o0, 64, %o0

	andcc		%o2, 0x30, %o3
	be,a		1f
	 srl		%g1, 24, %g2
4:	SMOVE_ALIGNCHUNK(o1, o0, 0x00, g2, g3, g4, g5, o4, o5, g6, g1, 24, 8, 3)
	subcc		%o3, 16, %o3
	add		%o1, 16, %o1
	bne		4b
	 add		%o0, 16, %o0

	srl		%g1, 24, %g2
1:	stb		%g2, [%o0 + 3]
	b		88f
	 add		%o0, 4, %o0
42:	SMOVE_ALIGNCHUNK(o1, o0, 0x00, g2, g3, g4, g5, o4, o5, g6, g1, 16, 16, -2)
	SMOVE_ALIGNCHUNK(o1, o0, 0x10, g2, g3, g4, g5, o4, o5, g6, g1, 16, 16, -2)
	SMOVE_ALIGNCHUNK(o1, o0, 0x20, g2, g3, g4, g5, o4, o5, g6, g1, 16, 16, -2)
	SMOVE_ALIGNCHUNK(o1, o0, 0x30, g2, g3, g4, g5, o4, o5, g6, g1, 16, 16, -2)
	subcc		%o3, 64, %o3
	add		%o1, 64, %o1
	bne		42b
	 add		%o0, 64, %o0

	andcc		%o2, 0x30, %o3
	be,a		1f
	 srl		%g1, 16, %g2
4:	SMOVE_ALIGNCHUNK(o1, o0, 0x00, g2, g3, g4, g5, o4, o5, g6, g1, 16, 16, -2)
	subcc		%o3, 16, %o3
	add		%o1, 16, %o1
	bne		4b
	 add		%o0, 16, %o0

	srl		%g1, 16, %g2
1:	sth		%g2, [%o0 - 2]

	/* Fall through */

88:	and		%o2, 0xe, %o3
	mov		%o7, %g2
	sll		%o3, 3, %o4
	add		%o0, %o3, %o0
106:	call		100f
	 add		%o1, %o3, %o1
	mov		%g2, %o7
	jmpl		%o5 + (89f - 106b), %g0
	 andcc		%o2, 1, %g0

	MOVE_SHORTCHUNK(o1, o0, 0x0c, g2, g3)
	MOVE_SHORTCHUNK(o1, o0, 0x0a, g2, g3)
	MOVE_SHORTCHUNK(o1, o0, 0x08, g2, g3)
	MOVE_SHORTCHUNK(o1, o0, 0x06, g2, g3)
	MOVE_SHORTCHUNK(o1, o0, 0x04, g2, g3)
	MOVE_SHORTCHUNK(o1, o0, 0x02, g2, g3)
	MOVE_SHORTCHUNK(o1, o0, 0x00, g2, g3)

89:	be		1f
	 nop

	ldub		[%o1], %g2
	stb		%g2, [%o0]
1:	retl
	 ld		[%sp + 64], %o0

90:	bne		88b
	 andcc		%o2, 8, %g0

	be		1f
	 andcc		%o2, 4, %g0

	ld		[%o1 + 0x00], %g2
	ld		[%o1 + 0x04], %g3
	add		%o1, 8, %o1
	st		%g2, [%o0 + 0x00]
	st		%g3, [%o0 + 0x04]
	add		%o0, 8, %o0
1:	b		81b
	 mov		%o2, %g1

100:	retl
	 sub		%o7, %o4, %o5
110:	retl
	 sub		%o7, %g6, %o5
END(memcpy)

libc_hidden_builtin_def (memcpy)

libc_hidden_def (__mempcpy)
weak_alias (__mempcpy, mempcpy)
libc_hidden_builtin_def (mempcpy)
