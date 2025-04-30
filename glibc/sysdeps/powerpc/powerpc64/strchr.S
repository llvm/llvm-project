/* Optimized strchr implementation for PowerPC64.
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

/* See strlen.s for comments on how this works.  */

/* char * [r3] strchr (const char *s [r3] , int c [r4] )  */

#ifndef STRCHR
# define STRCHR strchr
#endif

ENTRY_TOCLESS (STRCHR)
	CALL_MCOUNT 2

#define rTMP1	r0
#define rRTN	r3	/* outgoing result */
#define rSTR	r8	/* current word pointer */
#define rCHR	r4	/* byte we're looking for, spread over the whole word */
#define rWORD	r5	/* the current word */
#define rCLZB	rCHR	/* leading zero byte count */
#define rFEFE	r6	/* constant 0xfefefefefefefeff (-0x0101010101010101) */
#define r7F7F	r7	/* constant 0x7f7f7f7f7f7f7f7f */
#define rTMP2	r9
#define rIGN	r10	/* number of bits we should ignore in the first word */
#define rMASK	r11	/* mask with the bits to ignore set to 0 */
#define rTMP3	r12
#define rTMP4	rIGN
#define rTMP5	rMASK

	dcbt	0,rRTN
	insrdi	rCHR, rCHR, 8, 48
	li	rMASK, -1
	insrdi	rCHR, rCHR, 16, 32
	rlwinm	rIGN, rRTN, 3, 26, 28
	insrdi	rCHR, rCHR, 32, 0
	lis	rFEFE, -0x101
	lis	r7F7F, 0x7f7f
	clrrdi	rSTR, rRTN, 3
	addi	rFEFE, rFEFE, -0x101
	addi	r7F7F, r7F7F, 0x7f7f
	sldi	rTMP1, rFEFE, 32
	insrdi	r7F7F, r7F7F, 32, 0
	add	rFEFE, rFEFE, rTMP1
/* Test the first (partial?) word.  */
	ld	rWORD, 0(rSTR)
#ifdef __LITTLE_ENDIAN__
	sld	rMASK, rMASK, rIGN
#else
	srd	rMASK, rMASK, rIGN
#endif
	orc	rWORD, rWORD, rMASK
	add	rTMP1, rFEFE, rWORD
	nor	rTMP2, r7F7F, rWORD
	and.	rTMP4, rTMP1, rTMP2
	xor	rTMP3, rCHR, rWORD
	orc	rTMP3, rTMP3, rMASK
	b	L(loopentry)

/* The loop.  */

L(loop):
	ldu	rWORD, 8(rSTR)
	and.	rTMP5, rTMP1, rTMP2
/* Test for 0.	*/
	add	rTMP1, rFEFE, rWORD /* x - 0x01010101.  */
	nor	rTMP2, r7F7F, rWORD /* ~(x | 0x7f7f7f7f) == ~x & 0x80808080.  */
	bne	L(foundit)
	and.	rTMP4, rTMP1, rTMP2 /* (x - 0x01010101) & ~x & 0x80808080.  */
/* Start test for the bytes we're looking for.  */
	xor	rTMP3, rCHR, rWORD
L(loopentry):
	add	rTMP1, rFEFE, rTMP3
	nor	rTMP2, r7F7F, rTMP3
	beq	L(loop)

/* There is a zero byte in the word, but may also be a matching byte (either
   before or after the zero byte).  In fact, we may be looking for a
   zero byte, in which case we return a match.  */
	and.	rTMP5, rTMP1, rTMP2
	li	rRTN, 0
	beqlr
/* At this point:
   rTMP5 bytes are 0x80 for each match of c, 0 otherwise.
   rTMP4 bytes are 0x80 for each match of 0, 0 otherwise.
   But there may be false matches in the next most significant byte from
   a true match due to carries.  This means we need to recalculate the
   matches using a longer method for big-endian.  */
#ifdef __LITTLE_ENDIAN__
	addi	rTMP1, rTMP5, -1
	andc	rTMP1, rTMP1, rTMP5
	cntlzd	rCLZB, rTMP1
	addi	rTMP2, rTMP4, -1
	andc	rTMP2, rTMP2, rTMP4
	cmpld	rTMP1, rTMP2
	bgtlr
	subfic	rCLZB, rCLZB, 64-7
#else
/* I think we could reduce this by two instructions by keeping the "nor"
   results from the loop for reuse here.  See strlen.S tail.  Similarly
   one instruction could be pruned from L(foundit).  */
	and	rFEFE, r7F7F, rWORD
	or	rTMP5, r7F7F, rWORD
	and	rTMP1, r7F7F, rTMP3
	or	rTMP4, r7F7F, rTMP3
	add	rFEFE, rFEFE, r7F7F
	add	rTMP1, rTMP1, r7F7F
	nor	rWORD, rTMP5, rFEFE
	nor	rTMP2, rTMP4, rTMP1
	cntlzd	rCLZB, rTMP2
	cmpld	rWORD, rTMP2
	bgtlr
#endif
	srdi	rCLZB, rCLZB, 3
	add	rRTN, rSTR, rCLZB
	blr

L(foundit):
#ifdef __LITTLE_ENDIAN__
	addi	rTMP1, rTMP5, -1
	andc	rTMP1, rTMP1, rTMP5
	cntlzd	rCLZB, rTMP1
	subfic	rCLZB, rCLZB, 64-7-64
	sradi	rCLZB, rCLZB, 3
#else
	and	rTMP1, r7F7F, rTMP3
	or	rTMP4, r7F7F, rTMP3
	add	rTMP1, rTMP1, r7F7F
	nor	rTMP2, rTMP4, rTMP1
	cntlzd	rCLZB, rTMP2
	subi	rSTR, rSTR, 8
	srdi	rCLZB, rCLZB, 3
#endif
	add	rRTN, rSTR, rCLZB
	blr
END (STRCHR)

weak_alias (strchr, index)
libc_hidden_builtin_def (strchr)
