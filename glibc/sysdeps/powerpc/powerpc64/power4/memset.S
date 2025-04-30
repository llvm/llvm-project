/* Optimized memset implementation for PowerPC64.
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
	.machine power4
ENTRY_TOCLESS (MEMSET, 5)
	CALL_MCOUNT 3

#define rTMP	r0
#define rRTN	r3	/* Initial value of 1st argument.  */
#define rMEMP0	r3	/* Original value of 1st arg.  */
#define rCHR	r4	/* Char to set in each byte.  */
#define rLEN	r5	/* Length of region to set.  */
#define rMEMP	r6	/* Address at which we are storing.  */
#define rALIGN	r7	/* Number of bytes we are setting now (when aligning). */
#define rMEMP2	r8

#define rNEG64	r8	/* Constant -64 for clearing with dcbz.  */
#define rCLS	r8	/* Cache line size obtained from static.  */
#define rCLM	r9	/* Cache line size mask to check for cache alignment.  */
L(_memset):
/* Take care of case for size <= 4.  */
	cmpldi	cr1, rLEN, 8
	andi.	rALIGN, rMEMP0, 7
	mr	rMEMP, rMEMP0
	ble-	cr1, L(small)

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
L(caligned):
	cmpldi	cr1, rCHR, 0
	clrrdi.	rALIGN, rLEN, 5
	mtcrf	0x01, rLEN
	beq	cr1, L(zloopstart) /* Special case for clearing memory using dcbz.  */
L(nondcbz):
	srdi	rTMP, rALIGN, 5
	mtctr	rTMP
	beq	L(medium)	/* We may not actually get to do a full line.  */
	clrldi.	rLEN, rLEN, 59
	add	rMEMP, rMEMP, rALIGN
	li	rNEG64, -0x40
	bdz	L(cloopdone)

L(c3):	dcbtst	rNEG64, rMEMP
	std	rCHR, -8(rMEMP)
	std	rCHR, -16(rMEMP)
	std	rCHR, -24(rMEMP)
	stdu	rCHR, -32(rMEMP)
	bdnz	L(c3)
L(cloopdone):
	std	rCHR, -8(rMEMP)
	std	rCHR, -16(rMEMP)
	cmpldi	cr1, rLEN, 16
	std	rCHR, -24(rMEMP)
	stdu	rCHR, -32(rMEMP)
	beqlr
	add	rMEMP, rMEMP, rALIGN
	b	L(medium_tail2)

	.align 5
/* Clear lines of memory in 128-byte chunks.  */
L(zloopstart):
/* If the remaining length is less the 32 bytes, don't bother getting
	 the cache line size.  */
	beq	L(medium)
	li      rCLS,128  /* cache line size is 128 */

/* Now we know the cache line size, and it is not 32-bytes, but
	 we may not yet be aligned to the cache line. May have a partial
	 line to fill, so touch it 1st.  */
	dcbt	0,rMEMP
L(getCacheAligned):
	cmpldi	cr1,rLEN,32
	andi.	rTMP,rMEMP,127
	blt	cr1,L(handletail32)
	beq	L(cacheAligned)
	addi	rMEMP,rMEMP,32
	addi	rLEN,rLEN,-32
	std	rCHR,-32(rMEMP)
	std	rCHR,-24(rMEMP)
	std	rCHR,-16(rMEMP)
	std	rCHR,-8(rMEMP)
	b	L(getCacheAligned)

/* Now we are aligned to the cache line and can use dcbz.  */
L(cacheAligned):
	cmpld	cr1,rLEN,rCLS
	blt	cr1,L(handletail32)
	dcbz	0,rMEMP
	subf	rLEN,rCLS,rLEN
	add	rMEMP,rMEMP,rCLS
	b	L(cacheAligned)

/* We are here because the cache line size was set and was not 32-bytes
   and the remainder (rLEN) is less than the actual cache line size.
   So set up the preconditions for L(nondcbz) and go there.  */
L(handletail32):
	clrrwi.	rALIGN, rLEN, 5
	b		L(nondcbz)

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
	bt-	29, L(medium_29t)
L(medium_29f):
	bge-	cr1, L(medium_27t)
	bflr-	28
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
	blt-	cr1, L(medium_27f)
L(medium_27t):
	std	rCHR, -8(rMEMP)
	stdu	rCHR, -16(rMEMP)
L(medium_27f):
	bflr-	28
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
