/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Kazumoto Kojima <kkojima@rr.iij4u.or.jp>
   Optimized by Toshiyasu Morita <toshiyasu.morita@hsa.hitachi.com>

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

/* void *memcpy(void *dst, const void *src, size_t n);
    No overlap between the memory of DST and of SRC are assumed.  */

ENTRY(memcpy)
	mov	r4,r3		/* Save destination.  */

	/* If less than 11 bytes, just do a byte copy.  */
	mov	#11,r0
	cmp/gt	r6,r0
	bt	L_byteloop_init

	/* Check if we need to word-align source.  */
	mov	r5,r0
	tst	#1,r0
	bt	L_wordalign

	mov.b	@r0+,r1		/* Copy one byte.  */
	add	#-1,r6
	mov.b	r1,@r4
	add	#1,r4

	.balignw 4,0x0009
L_wordalign:
	/* Check if we need to longword-align source.  */
	tst	#2,r0
	bt	L_copy

	mov.w	@r0+,r1		/* Copy one word.  */
	add	#-2,r6
#ifdef __BIG_ENDIAN__
	add	#1,r4
	mov.b	r1,@r4
	shlr8	r1
	mov.b	r1,@-r4
	add	#2,r4
#else
	mov.b	r1,@r4
	add	#1,r4
	shlr8	r1
	mov.b	r1,@r4
	add	#1,r4
#endif
L_copy:
	mov	r0,r5

	/* Calculate the correct routine to handle the destination
	   alignment and simultaneously calculate the loop counts for
	   both the 2 word copy loop and byte copy loop.  */
	mova	L_jumptable,r0
	mov	r0,r1
	mov	r4,r0
	mov	r6,r7
	and	#3,r0
	shlr2	r7
	shll	r0
	shlr	r7
	mov.w	@(r0,r1),r2
	mov	#7,r0
	braf	r2
	and	r0,r6
L_base:

	.balign	4
L_jumptable:
	.word	L_copydest0 - L_base
	.word	L_copydest1_or_3 - L_base
	.word	L_copydest2 - L_base
	.word	L_copydest1_or_3 - L_base

	.balign	4
	/* Copy routine for (dest mod 4) == 1 or == 3.  */
L_copydest1_or_3:
	add	#-1,r4
	.balignw 4,0x0009
L_copydest1_or_3_loop:
	mov.l	@r5+,r0		/* Read first longword.  */
	dt	r7
	mov.l	@r5+,r1		/* Read second longword.  */
#ifdef __BIG_ENDIAN__
	/* Write first longword as byte, word, byte.  */
	mov.b	r0,@(4,r4)
	shlr8	r0
	mov.w	r0,@(2,r4)
	shlr16	r0
	mov.b	r0,@(1,r4)
	mov	r1,r0
	/* Write second longword as byte, word, byte.  */
	mov.b	r0,@(8,r4)
	shlr8	r0
	mov.w	r0,@(6,r4)
	shlr16	r0
	mov.b	r0,@(5,r4)
#else
	/* Write first longword as byte, word, byte.  */
	mov.b	r0,@(1,r4)
	shlr8	r0
	mov.w	r0,@(2,r4)
	shlr16	r0
	mov.b	r0,@(4,r4)
	mov	r1,r0
	/* Write second longword as byte, word, byte.  */
	mov.b	r0,@(5,r4)
	shlr8	r0
	mov.w	r0,@(6,r4)
	shlr16	r0
	mov.b	r0,@(8,r4)
#endif
	bf/s	L_copydest1_or_3_loop
	add	#8,r4

	bra	L_byteloop_init
	add	#1,r4

	.balign 4
	/* Copy routine for (dest mod 4) == 2.  */
L_copydest2:
L_copydest2_loop:
	mov.l	@r5+,r0
	dt	r7
	mov.l	@r5+,r1
#ifdef __BIG_ENDIAN__
	mov.w	r0,@(2,r4)
	shlr16	r0
	mov.w	r0,@r4
	mov	r1,r0
	mov.w	r0,@(6,r4)
	shlr16	r0
	mov.w	r0,@(4,r4)
#else
	mov.w	r0,@r4
	shlr16	r0
	mov.w	r0,@(2,r4)
	mov	r1,r0
	mov.w	r0,@(4,r4)
	shlr16	r0
	mov.w	r0,@(6,r4)
#endif
	bf/s	L_copydest2_loop
	add	#8,r4

	bra	L_byteloop_init
	nop

	.balign 4
	/* Copy routine for (dest mod 4) == 0.  */
L_copydest0:
	add	#-8,r4
	.balignw 4,0x0009
L_copydest0_loop:
	mov.l	@r5+,r0
	dt	r7
	mov.l	@r5+,r1
	add	#8,r4
	mov.l	r0,@r4
	bf/s	L_copydest0_loop
	mov.l	r1,@(4,r4)

	add	#8,r4		/* Fall through.  */

L_byteloop_init:
	tst	r6,r6
	bt	L_exit

	.balignw 4,0x0009
	/* Copy remaining bytes.  */
L_byteloop:
	mov.b	@r5+,r0
	dt	r6
	mov.b	r0,@r4
	bf/s	L_byteloop
	add	#1,r4

L_exit:
	rts
	mov	r3,r0		/* Return destination.  */
END(memcpy)
libc_hidden_builtin_def (memcpy)
