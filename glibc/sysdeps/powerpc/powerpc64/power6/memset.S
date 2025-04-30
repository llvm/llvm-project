/* Optimized 64-bit memset implementation for POWER6.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

/* void * [r3] memset (void *s [r3], int c [r4], size_t n [r5]));
   Returns 's'.

   The memset is done in three sizes: byte (8 bits), word (32 bits),
   cache line (256 bits). There is a special case for setting cache lines
   to 0, to take advantage of the dcbz instruction.  */

#ifndef MEMSET
# define MEMSET memset
#endif
	.machine power6
ENTRY_TOCLESS (MEMSET, 7)
	CALL_MCOUNT 3

#define rTMP	r0
#define rRTN	r3	/* Initial value of 1st argument.  */
#define rMEMP0	r3	/* Original value of 1st arg.  */
#define rCHR	r4	/* Char to set in each byte.  */
#define rLEN	r5	/* Length of region to set.  */
#define rMEMP	r6	/* Address at which we are storing.  */
#define rALIGN	r7	/* Number of bytes we are setting now (when aligning). */
#define rMEMP2	r8
#define rMEMP3	r9	/* Alt mem pointer.  */
L(_memset):
/* Take care of case for size <= 4.  */
	cmpldi	cr1, rLEN, 8
	andi.	rALIGN, rMEMP0, 7
	mr	rMEMP, rMEMP0
	ble	cr1, L(small)

/* Align to doubleword boundary.  */
	cmpldi	cr5, rLEN, 31
	insrdi	rCHR, rCHR, 8, 48	/* Replicate byte to halfword.  */
	beq+	L(aligned2)
	mtcrf	0x01, rMEMP0
	subfic	rALIGN, rALIGN, 8
	cror	28,30,31		/* Detect odd word aligned.  */
	add	rMEMP, rMEMP, rALIGN
	sub	rLEN, rLEN, rALIGN
	insrdi	rCHR, rCHR, 16, 32	/* Replicate halfword to word.  */
	bt	29, L(g4)
/* Process the even word of doubleword.  */
	bf+	31, L(g2)
	stb	rCHR, 0(rMEMP0)
	bt	30, L(g4x)
L(g2):
	sth	rCHR, -6(rMEMP)
L(g4x):
	stw	rCHR, -4(rMEMP)
	b	L(aligned)
/* Process the odd word of doubleword.  */
L(g4):
	bf	28, L(g4x) /* If false, word aligned on odd word.  */
	bf+	31, L(g0)
	stb	rCHR, 0(rMEMP0)
	bt	30, L(aligned)
L(g0):
	sth	rCHR, -2(rMEMP)

/* Handle the case of size < 31.  */
L(aligned2):
	insrdi	rCHR, rCHR, 16, 32	/* Replicate halfword to word.  */
L(aligned):
	mtcrf	0x01, rLEN
	ble	cr5, L(medium)
/* Align to 32-byte boundary.  */
	andi.	rALIGN, rMEMP, 0x18
	subfic	rALIGN, rALIGN, 0x20
	insrdi	rCHR, rCHR, 32, 0	/* Replicate word to double word. */
	beq	L(caligned)
	mtcrf	0x01, rALIGN
	add	rMEMP, rMEMP, rALIGN
	sub	rLEN, rLEN, rALIGN
	cmplwi	cr1, rALIGN, 0x10
	mr	rMEMP2, rMEMP
	bf	28, L(a1)
	stdu	rCHR, -8(rMEMP2)
L(a1):	blt	cr1, L(a2)
	std	rCHR, -8(rMEMP2)
	stdu	rCHR, -16(rMEMP2)
L(a2):

/* Now aligned to a 32 byte boundary.  */
        .align 4
L(caligned):
	cmpldi	cr1, rCHR, 0
	clrrdi.	rALIGN, rLEN, 5
	mtcrf	0x01, rLEN
	beq	cr1, L(zloopstart) /* Special case for clearing memory using dcbz.  */
	beq	L(medium)	/* We may not actually get to do a full line.  */
	.align 4
/* Storing a non-zero "c" value. We are aligned at a sector (32-byte)
   boundary may not be at cache line (128-byte) boundary.  */
L(nzloopstart):
/* memset in 32-byte chunks until we get to a cache line boundary.
   If rLEN is less than the distance to the next cache-line boundary use
   cacheAligned1 code to finish the tail.  */
	cmpldi	cr1,rLEN,128

	andi.	rTMP,rMEMP,127
	blt	cr1,L(cacheAligned1)
	addi	rMEMP3,rMEMP,32
	beq	L(nzCacheAligned)
	addi	rLEN,rLEN,-32
	std	rCHR,0(rMEMP)
	std	rCHR,8(rMEMP)
	std	rCHR,16(rMEMP)
	addi	rMEMP,rMEMP,32
	andi.	rTMP,rMEMP3,127
	std	rCHR,-8(rMEMP3)

	beq	L(nzCacheAligned)
	addi	rLEN,rLEN,-32
	std	rCHR,0(rMEMP3)
	addi	rMEMP,rMEMP,32
	std	rCHR,8(rMEMP3)
	andi.	rTMP,rMEMP,127
	std	rCHR,16(rMEMP3)
	std	rCHR,24(rMEMP3)

	beq	L(nzCacheAligned)
	addi	rLEN,rLEN,-32
	std	rCHR,32(rMEMP3)
	addi	rMEMP,rMEMP,32
	cmpldi	cr1,rLEN,128
	std	rCHR,40(rMEMP3)
	cmpldi	cr6,rLEN,256
	li	rMEMP2,128
	std	rCHR,48(rMEMP3)
	std	rCHR,56(rMEMP3)
	blt	cr1,L(cacheAligned1)
	b	L(nzCacheAligned128)

/* Now we are aligned to the cache line and can use dcbtst.  */
        .align 4
L(nzCacheAligned):
	cmpldi	cr1,rLEN,128
	blt	cr1,L(cacheAligned1)
	b	L(nzCacheAligned128)
        .align 5
L(nzCacheAligned128):
	cmpldi	cr1,rLEN,256
	addi	rMEMP3,rMEMP,64
	std	rCHR,0(rMEMP)
	std	rCHR,8(rMEMP)
	std	rCHR,16(rMEMP)
	std	rCHR,24(rMEMP)
	std	rCHR,32(rMEMP)
	std	rCHR,40(rMEMP)
	std	rCHR,48(rMEMP)
	std	rCHR,56(rMEMP)
	addi	rMEMP,rMEMP3,64
	addi	rLEN,rLEN,-128
	std	rCHR,0(rMEMP3)
	std	rCHR,8(rMEMP3)
	std	rCHR,16(rMEMP3)
	std	rCHR,24(rMEMP3)
	std	rCHR,32(rMEMP3)
	std	rCHR,40(rMEMP3)
	std	rCHR,48(rMEMP3)
	std	rCHR,56(rMEMP3)
	bge	cr1,L(nzCacheAligned128)
	dcbtst	0,rMEMP
	b	L(cacheAligned1)
	.align 5
/* Storing a zero "c" value. We are aligned at a sector (32-byte)
   boundary but may not be at cache line (128-byte) boundary.  If the
   remaining length spans a full cache line we can use the Data cache
   block zero instruction. */
L(zloopstart):
/* memset in 32-byte chunks until we get to a cache line boundary.
   If rLEN is less than the distance to the next cache-line boundary use
   cacheAligned1 code to finish the tail.  */
	cmpldi	cr1,rLEN,128
	beq	L(medium)
L(getCacheAligned):
	andi.	rTMP,rMEMP,127
	nop
	blt	cr1,L(cacheAligned1)
	addi	rMEMP3,rMEMP,32
	beq	L(cacheAligned)
	addi	rLEN,rLEN,-32
	std	rCHR,0(rMEMP)
	std	rCHR,8(rMEMP)
	std	rCHR,16(rMEMP)
	addi	rMEMP,rMEMP,32
	andi.	rTMP,rMEMP3,127
	std	rCHR,-8(rMEMP3)
L(getCacheAligned2):
	beq	L(cacheAligned)
	addi	rLEN,rLEN,-32
	std	rCHR,0(rMEMP3)
	std	rCHR,8(rMEMP3)
	addi	rMEMP,rMEMP,32
	andi.	rTMP,rMEMP,127
	std	rCHR,16(rMEMP3)
	std	rCHR,24(rMEMP3)
L(getCacheAligned3):
	beq	L(cacheAligned)
	addi	rLEN,rLEN,-32
	std	rCHR,32(rMEMP3)
	addi	rMEMP,rMEMP,32
	cmpldi	cr1,rLEN,128
	std	rCHR,40(rMEMP3)
	cmpldi	cr6,rLEN,256
	li	rMEMP2,128
	std	rCHR,48(rMEMP3)
	std	rCHR,56(rMEMP3)
	blt	cr1,L(cacheAligned1)
	blt	cr6,L(cacheAligned128)
	b	L(cacheAlignedx)

/* Now we are aligned to the cache line and can use dcbz.  */
        .align 5
L(cacheAligned):
	cmpldi	cr1,rLEN,128
	cmpldi	cr6,rLEN,256
	blt	cr1,L(cacheAligned1)
	li	rMEMP2,128
L(cacheAlignedx):
	cmpldi	cr5,rLEN,640
	blt	cr6,L(cacheAligned128)
	bgt	cr5,L(cacheAligned512)
	cmpldi	cr6,rLEN,512
	dcbz	0,rMEMP
	cmpldi	cr1,rLEN,384
	dcbz	rMEMP2,rMEMP
	addi	rMEMP,rMEMP,256
	addi	rLEN,rLEN,-256
	blt	cr1,L(cacheAligned1)
	blt	cr6,L(cacheAligned128)
	b	L(cacheAligned256)
	.align 5
/* A simple loop for the longer (>640 bytes) lengths.  This form limits
   the branch miss-predicted to exactly 1 at loop exit.*/
L(cacheAligned512):
	cmpldi	cr1,rLEN,128
	blt	cr1,L(cacheAligned1)
	dcbz	0,rMEMP
	addi	rLEN,rLEN,-128
	addi	rMEMP,rMEMP,128
	b	L(cacheAligned512)
        .align 5
L(cacheAligned256):

	cmpldi	cr6,rLEN,512

	dcbz	0,rMEMP
	cmpldi	cr1,rLEN,384
	dcbz	rMEMP2,rMEMP
	addi	rMEMP,rMEMP,256
	addi	rLEN,rLEN,-256

	bge	cr6,L(cacheAligned256)

	blt	cr1,L(cacheAligned1)
        .align 4
L(cacheAligned128):
	dcbz	0,rMEMP
	addi	rMEMP,rMEMP,128
	addi	rLEN,rLEN,-128
        nop
L(cacheAligned1):
	cmpldi	cr1,rLEN,32
	blt	cr1,L(handletail32)
	addi	rMEMP3,rMEMP,32
	addi	rLEN,rLEN,-32
	std	rCHR,0(rMEMP)
	std	rCHR,8(rMEMP)
	std	rCHR,16(rMEMP)
	addi	rMEMP,rMEMP,32
	cmpldi	cr1,rLEN,32
	std	rCHR,-8(rMEMP3)
L(cacheAligned2):
	blt	cr1,L(handletail32)
	addi	rLEN,rLEN,-32
	std	rCHR,0(rMEMP3)
	std	rCHR,8(rMEMP3)
	addi	rMEMP,rMEMP,32
	cmpldi	cr1,rLEN,32
	std	rCHR,16(rMEMP3)
	std	rCHR,24(rMEMP3)
	nop
L(cacheAligned3):
	blt	cr1,L(handletail32)
	addi	rMEMP,rMEMP,32
	addi	rLEN,rLEN,-32
	std	rCHR,32(rMEMP3)
	std	rCHR,40(rMEMP3)
	std	rCHR,48(rMEMP3)
	std	rCHR,56(rMEMP3)

/* We are here because the length or remainder (rLEN) is less than the
   cache line/sector size and does not justify aggressive loop unrolling.
   So set up the preconditions for L(medium) and go there.  */
        .align 3
L(handletail32):
	cmpldi	cr1,rLEN,0
	beqlr   cr1
	b	L(medium)

	.align 5
L(small):
/* Memset of 8 bytes or less.  */
	cmpldi	cr6, rLEN, 4
	cmpldi	cr5, rLEN, 1
	ble	cr6,L(le4)
	subi	rLEN, rLEN, 4
	stb	rCHR,0(rMEMP)
	stb	rCHR,1(rMEMP)
	stb	rCHR,2(rMEMP)
	stb	rCHR,3(rMEMP)
	addi	rMEMP,rMEMP, 4
	cmpldi	cr5, rLEN, 1
L(le4):
	cmpldi	cr1, rLEN, 3
	bltlr	cr5
	stb	rCHR, 0(rMEMP)
	beqlr	cr5
	stb	rCHR, 1(rMEMP)
	bltlr	cr1
	stb	rCHR, 2(rMEMP)
	beqlr	cr1
	stb	rCHR, 3(rMEMP)
	blr

/* Memset of 0-31 bytes.  */
	.align 5
L(medium):
	insrdi	rCHR, rCHR, 32, 0	/* Replicate word to double word.  */
	cmpldi	cr1, rLEN, 16
L(medium_tail2):
	add	rMEMP, rMEMP, rLEN
L(medium_tail):
	bt-	31, L(medium_31t)
	bt-	30, L(medium_30t)
L(medium_30f):
	bt	29, L(medium_29t)
L(medium_29f):
	bge	cr1, L(medium_27t)
	bflr	28
	std	rCHR, -8(rMEMP)
	blr

L(medium_31t):
	stbu	rCHR, -1(rMEMP)
	bf-	30, L(medium_30f)
L(medium_30t):
	sthu	rCHR, -2(rMEMP)
	bf-	29, L(medium_29f)
L(medium_29t):
	stwu	rCHR, -4(rMEMP)
	blt	cr1, L(medium_27f)
L(medium_27t):
	std	rCHR, -8(rMEMP)
	stdu	rCHR, -16(rMEMP)
L(medium_27f):
	bflr	28
L(medium_28t):
	std	rCHR, -8(rMEMP)
	blr
END_GEN_TB (MEMSET,TB_TOCLESS)
libc_hidden_builtin_def (memset)

/* Copied from bzero.S to prevent the linker from inserting a stub
   between bzero and memset.  */
ENTRY_TOCLESS (__bzero)
	CALL_MCOUNT 3
	mr	r5,r4
	li	r4,0
	b	L(_memset)
END (__bzero)
#ifndef __bzero
weak_alias (__bzero, bzero)
#endif
