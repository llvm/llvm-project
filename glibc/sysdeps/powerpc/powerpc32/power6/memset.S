/* Optimized 32-bit memset implementation for POWER6.
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
   cache line (1024 bits). There is a special case for setting cache lines
   to 0, to take advantage of the dcbz instruction.  */

	.machine power6
EALIGN (memset, 7, 0)
	CALL_MCOUNT

#define rTMP	r0
#define rRTN	r3	/* Initial value of 1st argument.  */
#define rMEMP0	r3	/* Original value of 1st arg.  */
#define rCHR	r4	/* Char to set in each byte.  */
#define rLEN	r5	/* Length of region to set.  */
#define rMEMP	r6	/* Address at which we are storing.  */
#define rALIGN	r7	/* Number of bytes we are setting now (when aligning). */
#define rMEMP2	r8

#define rNEG64	r8	/* Constant -64 for clearing with dcbz.  */
#define rMEMP3	r9	/* Alt mem pointer.  */
L(_memset):
/* Take care of case for size <= 4.  */
	cmplwi	cr1, rLEN, 4
	andi.	rALIGN, rMEMP0, 3
	mr	rMEMP, rMEMP0
	ble-	cr1, L(small)
/* Align to word boundary.  */
	cmplwi	cr5, rLEN, 31
	insrwi	rCHR, rCHR, 8, 16	/* Replicate byte to halfword.  */
	beq+	L(aligned)
	mtcrf	0x01, rMEMP0
	subfic	rALIGN, rALIGN, 4
	add	rMEMP, rMEMP, rALIGN
	sub	rLEN, rLEN, rALIGN
	bf+	31, L(g0)
	stb	rCHR, 0(rMEMP0)
	bt	30, L(aligned)
L(g0):
	sth	rCHR, -2(rMEMP)

        .align 4
/* Handle the case of size < 31.  */
L(aligned):
	mtcrf	0x01, rLEN
	insrwi	rCHR, rCHR, 16, 0	/* Replicate halfword to word.  */
	ble	cr5, L(medium)
/* Align to 32-byte boundary.  */
	andi.	rALIGN, rMEMP, 0x1C
	subfic	rALIGN, rALIGN, 0x20
	beq	L(caligned)
	mtcrf	0x01, rALIGN
	add	rMEMP, rMEMP, rALIGN
	sub	rLEN, rLEN, rALIGN
	cmplwi	cr1, rALIGN, 0x10
	mr	rMEMP2, rMEMP
	bf	28, L(a1)
        stw     rCHR, -4(rMEMP2)
	stwu	rCHR, -8(rMEMP2)
	nop
L(a1):	blt	cr1, L(a2)
        stw     rCHR, -4(rMEMP2)
	stw	rCHR, -8(rMEMP2)
	stw	rCHR, -12(rMEMP2)
	stwu	rCHR, -16(rMEMP2)
L(a2):  bf      29, L(caligned)
        stw     rCHR, -4(rMEMP2)

        .align 3
/* Now aligned to a 32 byte boundary.  */
L(caligned):
	cmplwi	cr1, rCHR, 0
	clrrwi.	rALIGN, rLEN, 5
	mtcrf	0x01, rLEN
	beq	cr1, L(zloopstart) /* Special case for clearing memory using dcbz.  */
L(nondcbz):
	beq	L(medium)	/* We may not actually get to do a full line.  */
	nop
/* Storing a non-zero "c" value. We are aligned at a sector (32-byte)
   boundary may not be at cache line (128-byte) boundary.  */
L(nzloopstart):
/* memset in 32-byte chunks until we get to a cache line boundary.
   If rLEN is less than the distance to the next cache-line boundary use
   cacheAligned1 code to finish the tail.  */
	cmplwi	cr1,rLEN,128

	andi.	rTMP,rMEMP,127
	blt	cr1,L(cacheAligned1)
	addi	rMEMP3,rMEMP,32
	beq	L(nzCacheAligned)
	addi	rLEN,rLEN,-32
	stw	rCHR,0(rMEMP)
        stw     rCHR,4(rMEMP)
	stw	rCHR,8(rMEMP)
	stw     rCHR,12(rMEMP)
	stw	rCHR,16(rMEMP)
        stw     rCHR,20(rMEMP)
	addi	rMEMP,rMEMP,32
	andi.	rTMP,rMEMP3,127
	stw	rCHR,-8(rMEMP3)
        stw     rCHR,-4(rMEMP3)

	beq	L(nzCacheAligned)
	addi	rLEN,rLEN,-32
	stw	rCHR,0(rMEMP3)
        stw     rCHR,4(rMEMP3)
	addi	rMEMP,rMEMP,32
	stw	rCHR,8(rMEMP3)
	stw     rCHR,12(rMEMP3)
	andi.	rTMP,rMEMP,127
	stw	rCHR,16(rMEMP3)
        stw     rCHR,20(rMEMP3)
	stw	rCHR,24(rMEMP3)
        stw     rCHR,28(rMEMP3)

	beq	L(nzCacheAligned)
	addi	rLEN,rLEN,-32
/* At this point we can overrun the store queue (pipe reject) so it is
   time to slow things down. The store queue can merge two adjacent
   stores into a single L1/L2 op, but the L2 is clocked at 1/2 the CPU.
   So we add "group ending nops" to guarantee that we dispatch only two
   stores every other cycle. */
	ori	r1,r1,0
	ori	r1,r1,0
	stw	rCHR,32(rMEMP3)
        stw     rCHR,36(rMEMP3)
	addi	rMEMP,rMEMP,32
	cmplwi	cr1,rLEN,128
	ori	r1,r1,0
	ori	r1,r1,0
	stw	rCHR,40(rMEMP3)
	stw     rCHR,44(rMEMP3)
	ori	r1,r1,0
	ori	r1,r1,0
	stw	rCHR,48(rMEMP3)
        stw     rCHR,52(rMEMP3)
	ori	r1,r1,0
	ori	r1,r1,0
	stw	rCHR,56(rMEMP3)
        stw     rCHR,60(rMEMP3)
	blt	cr1,L(cacheAligned1)
	b	L(nzCacheAligned)

/* Now we are aligned to the cache line and can use dcbtst.  */
        .align 5
L(nzCacheAligned):
	cmplwi	cr1,rLEN,128
	cmplwi	cr6,rLEN,256
	blt	cr1,L(cacheAligned1)
	blt	cr6,L(nzCacheAligned128)
        .align 4
L(nzCacheAligned128):
	nop
	addi	rMEMP3,rMEMP,64
	stw	rCHR,0(rMEMP)
        stw     rCHR,4(rMEMP)
	stw	rCHR,8(rMEMP)
	stw     rCHR,12(rMEMP)
	stw	rCHR,16(rMEMP)
        stw     rCHR,20(rMEMP)
	stw	rCHR,24(rMEMP)
        stw     rCHR,28(rMEMP)
	stw	rCHR,32(rMEMP)
        stw     rCHR,36(rMEMP)
	stw	rCHR,40(rMEMP)
	stw     rCHR,44(rMEMP)
	stw	rCHR,48(rMEMP)
        stw     rCHR,52(rMEMP)
	stw	rCHR,56(rMEMP)
        stw     rCHR,60(rMEMP)
	addi	rMEMP,rMEMP3,64
	addi	rLEN,rLEN,-128
/* At this point we can overrun the store queue (pipe reject) so it is
   time to slow things down. The store queue can merge two adjacent
   stores into a single L1/L2 op, but the L2 is clocked at 1/2 the CPU.
   So we add "group ending nops" to guarantee that we dispatch only one
   store per cycle. */
	stw	rCHR,0(rMEMP3)
	ori	r1,r1,0
        stw     rCHR,4(rMEMP3)
	ori	r1,r1,0
	stw	rCHR,8(rMEMP3)
	ori	r1,r1,0
	stw     rCHR,12(rMEMP3)
	ori	r1,r1,0
	stw	rCHR,16(rMEMP3)
	ori	r1,r1,0
        stw     rCHR,20(rMEMP3)
	ori	r1,r1,0
	stw	rCHR,24(rMEMP3)
	ori	r1,r1,0
        stw     rCHR,28(rMEMP3)
	ori	r1,r1,0
	stw	rCHR,32(rMEMP3)
	ori	r1,r1,0
        stw     rCHR,36(rMEMP3)
	ori	r1,r1,0
	stw	rCHR,40(rMEMP3)
	ori	r1,r1,0
	stw     rCHR,44(rMEMP3)
	ori	r1,r1,0
	stw	rCHR,48(rMEMP3)
	ori	r1,r1,0
        stw     rCHR,52(rMEMP3)
	ori	r1,r1,0
	stw	rCHR,56(rMEMP3)
	ori	r1,r1,0
        stw     rCHR,60(rMEMP3)
	blt	cr6,L(cacheAligned1)
#if IS_IN (libc)
	lfd	0,-128(rMEMP)
#endif
	b	L(nzCacheAligned256)
        .align 5
L(nzCacheAligned256):
	cmplwi	cr1,rLEN,256
	addi	rMEMP3,rMEMP,64
#if !IS_IN (libc)
/* When we are not in libc we should use only GPRs to avoid the FPU lock
   interrupt.  */
	stw	rCHR,0(rMEMP)
        stw     rCHR,4(rMEMP)
	stw	rCHR,8(rMEMP)
	stw     rCHR,12(rMEMP)
	stw	rCHR,16(rMEMP)
        stw     rCHR,20(rMEMP)
	stw	rCHR,24(rMEMP)
        stw     rCHR,28(rMEMP)
	stw	rCHR,32(rMEMP)
        stw     rCHR,36(rMEMP)
	stw	rCHR,40(rMEMP)
	stw     rCHR,44(rMEMP)
	stw	rCHR,48(rMEMP)
        stw     rCHR,52(rMEMP)
	stw	rCHR,56(rMEMP)
        stw     rCHR,60(rMEMP)
	addi	rMEMP,rMEMP3,64
	addi	rLEN,rLEN,-128
	stw	rCHR,0(rMEMP3)
        stw     rCHR,4(rMEMP3)
	stw	rCHR,8(rMEMP3)
	stw     rCHR,12(rMEMP3)
	stw	rCHR,16(rMEMP3)
        stw     rCHR,20(rMEMP3)
	stw	rCHR,24(rMEMP3)
        stw     rCHR,28(rMEMP3)
	stw	rCHR,32(rMEMP3)
        stw     rCHR,36(rMEMP3)
	stw	rCHR,40(rMEMP3)
	stw     rCHR,44(rMEMP3)
	stw	rCHR,48(rMEMP3)
        stw     rCHR,52(rMEMP3)
	stw	rCHR,56(rMEMP3)
        stw     rCHR,60(rMEMP3)
#else
/* We are in libc and this is a long memset so we can use FPRs and can afford
   occasional FPU locked interrupts.  */
	stfd	0,0(rMEMP)
	stfd	0,8(rMEMP)
	stfd	0,16(rMEMP)
	stfd	0,24(rMEMP)
	stfd	0,32(rMEMP)
	stfd	0,40(rMEMP)
	stfd	0,48(rMEMP)
	stfd	0,56(rMEMP)
	addi	rMEMP,rMEMP3,64
	addi	rLEN,rLEN,-128
	stfd	0,0(rMEMP3)
	stfd	0,8(rMEMP3)
	stfd	0,16(rMEMP3)
	stfd	0,24(rMEMP3)
	stfd	0,32(rMEMP3)
	stfd	0,40(rMEMP3)
	stfd	0,48(rMEMP3)
	stfd	0,56(rMEMP3)
#endif
	bge	cr1,L(nzCacheAligned256)
	dcbtst	0,rMEMP
	b	L(cacheAligned1)

	.align 4
/* Storing a zero "c" value. We are aligned at a sector (32-byte)
   boundary but may not be at cache line (128-byte) boundary.  If the
   remaining length spans a full cache line we can use the Data cache
   block zero instruction. */
L(zloopstart):
/* memset in 32-byte chunks until we get to a cache line boundary.
   If rLEN is less than the distance to the next cache-line boundary use
   cacheAligned1 code to finish the tail.  */
	cmplwi	cr1,rLEN,128
	beq	L(medium)
L(getCacheAligned):
	andi.	rTMP,rMEMP,127
	blt	cr1,L(cacheAligned1)
	addi	rMEMP3,rMEMP,32
	beq	L(cacheAligned)
	addi	rLEN,rLEN,-32
	stw	rCHR,0(rMEMP)
        stw     rCHR,4(rMEMP)
	stw	rCHR,8(rMEMP)
	stw     rCHR,12(rMEMP)
	stw	rCHR,16(rMEMP)
        stw     rCHR,20(rMEMP)
	addi	rMEMP,rMEMP,32
	andi.	rTMP,rMEMP3,127
	stw	rCHR,-8(rMEMP3)
        stw     rCHR,-4(rMEMP3)
L(getCacheAligned2):
	beq	L(cacheAligned)
	addi	rLEN,rLEN,-32
	addi	rMEMP,rMEMP,32
	stw	rCHR,0(rMEMP3)
        stw     rCHR,4(rMEMP3)
	stw	rCHR,8(rMEMP3)
	stw     rCHR,12(rMEMP3)
	andi.	rTMP,rMEMP,127
	nop
	stw	rCHR,16(rMEMP3)
        stw     rCHR,20(rMEMP3)
	stw	rCHR,24(rMEMP3)
        stw     rCHR,28(rMEMP3)
L(getCacheAligned3):
	beq	L(cacheAligned)
/* At this point we can overrun the store queue (pipe reject) so it is
   time to slow things down. The store queue can merge two adjacent
   stores into a single L1/L2 op, but the L2 is clocked at 1/2 the CPU.
   So we add "group ending nops" to guarantee that we dispatch only two
   stores every other cycle. */
	addi	rLEN,rLEN,-32
	ori	r1,r1,0
	ori	r1,r1,0
	stw	rCHR,32(rMEMP3)
        stw     rCHR,36(rMEMP3)
	addi	rMEMP,rMEMP,32
	cmplwi	cr1,rLEN,128
	ori	r1,r1,0
	stw	rCHR,40(rMEMP3)
	stw     rCHR,44(rMEMP3)
	cmplwi	cr6,rLEN,256
	li	rMEMP2,128
	ori	r1,r1,0
	stw	rCHR,48(rMEMP3)
        stw     rCHR,52(rMEMP3)
	ori	r1,r1,0
	ori	r1,r1,0
	stw	rCHR,56(rMEMP3)
        stw     rCHR,60(rMEMP3)
	blt	cr1,L(cacheAligned1)
	blt	cr6,L(cacheAligned128)
	b	L(cacheAlignedx)

/* Now we are aligned to the cache line and can use dcbz.  */
        .align 4
L(cacheAligned):
	cmplwi	cr1,rLEN,128
	cmplwi	cr6,rLEN,256
	blt	cr1,L(cacheAligned1)
	li	rMEMP2,128
L(cacheAlignedx):
	cmplwi	cr5,rLEN,640
	blt	cr6,L(cacheAligned128)
	bgt	cr5,L(cacheAligned512)
	cmplwi	cr6,rLEN,512
	dcbz	0,rMEMP
	cmplwi	cr1,rLEN,384
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
	cmplwi	cr1,rLEN,128
	blt	cr1,L(cacheAligned1)
	dcbz	0,rMEMP
	addi	rLEN,rLEN,-128
	addi	rMEMP,rMEMP,128
	b	L(cacheAligned512)
        .align 5
L(cacheAligned256):
	cmplwi	cr6,rLEN,512
	dcbz	0,rMEMP
	cmplwi	cr1,rLEN,384
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
        .align 4
L(cacheAligned1):
	cmplwi	cr1,rLEN,32
	blt	cr1,L(handletail32)
	addi	rMEMP3,rMEMP,32
	addi	rLEN,rLEN,-32
	stw	rCHR,0(rMEMP)
        stw     rCHR,4(rMEMP)
	stw	rCHR,8(rMEMP)
	stw     rCHR,12(rMEMP)
	stw	rCHR,16(rMEMP)
        stw     rCHR,20(rMEMP)
	addi	rMEMP,rMEMP,32
	cmplwi	cr1,rLEN,32
	stw	rCHR,-8(rMEMP3)
        stw     rCHR,-4(rMEMP3)
L(cacheAligned2):
	blt	cr1,L(handletail32)
	addi	rLEN,rLEN,-32
	stw	rCHR,0(rMEMP3)
        stw     rCHR,4(rMEMP3)
	stw	rCHR,8(rMEMP3)
	stw     rCHR,12(rMEMP3)
	addi	rMEMP,rMEMP,32
	cmplwi	cr1,rLEN,32
	stw	rCHR,16(rMEMP3)
        stw     rCHR,20(rMEMP3)
	stw	rCHR,24(rMEMP3)
        stw     rCHR,28(rMEMP3)
	nop
L(cacheAligned3):
	blt	cr1,L(handletail32)
/* At this point we can overrun the store queue (pipe reject) so it is
   time to slow things down. The store queue can merge two adjacent
   stores into a single L1/L2 op, but the L2 is clocked at 1/2 the CPU.
   So we add "group ending nops" to guarantee that we dispatch only two
   stores every other cycle. */
	ori	r1,r1,0
	ori	r1,r1,0
	addi	rMEMP,rMEMP,32
	addi	rLEN,rLEN,-32
	ori	r1,r1,0
	ori	r1,r1,0
	stw	rCHR,32(rMEMP3)
        stw     rCHR,36(rMEMP3)
	ori	r1,r1,0
	ori	r1,r1,0
	stw	rCHR,40(rMEMP3)
	stw     rCHR,44(rMEMP3)
	ori	r1,r1,0
	ori	r1,r1,0
	stw	rCHR,48(rMEMP3)
        stw     rCHR,52(rMEMP3)
	ori	r1,r1,0
	ori	r1,r1,0
	stw	rCHR,56(rMEMP3)
        stw     rCHR,60(rMEMP3)

/* We are here because the length or remainder (rLEN) is less than the
   cache line/sector size and does not justify aggressive loop unrolling.
   So set up the preconditions for L(medium) and go there.  */
        .align 3
L(handletail32):
	cmplwi	cr1,rLEN,0
	beqlr   cr1
	b	L(medium)

	.align 4
L(small):
/* Memset of 4 bytes or less.  */
	cmplwi	cr5, rLEN, 1
	cmplwi	cr1, rLEN, 3
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
	cmplwi	cr1, rLEN, 16
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
        stw     rCHR, -4(rMEMP)
	stw	rCHR, -8(rMEMP)
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
        stw     rCHR, -4(rMEMP)
	stw	rCHR, -8(rMEMP)
        stw     rCHR, -12(rMEMP)
	stwu	rCHR, -16(rMEMP)
L(medium_27f):
	bflr	28
L(medium_28t):
        stw     rCHR, -4(rMEMP)
	stw	rCHR, -8(rMEMP)
	blr
END (memset)
libc_hidden_builtin_def (memset)
