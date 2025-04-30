/* Optimized memset implementation for PowerPC32/POWER7.
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
   Contributed by Luis Machado <luisgpm@br.ibm.com>.
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
   Returns 's'.  */

	.machine  power7
EALIGN (memset, 5, 0)
	CALL_MCOUNT

	.align	4
L(_memset):
	cmplwi	cr7,5,31
	cmplwi	cr6,5,8
	mr	10,3		/* Save original argument for later.  */
	mr	7,1		/* Save original r1 for later.  */
	cfi_offset(31,-8)

	/* Replicate byte to word.  */
	insrwi	4,4,8,16
	insrwi	4,4,16,0

	ble	cr6,L(small)	/* If length <= 8, use short copy code.  */

	neg	0,3
	ble	cr7,L(medium)	/* If length < 32, use medium copy code.  */

	/* Save our word twice to create a doubleword that we will later
	   copy to a FPR.  */
	stwu	1,-32(1)
	andi.	11,10,7		/* Check alignment of DST.  */
	mr	12,5
	stw	4,24(1)
	stw	4,28(1)
	beq	L(big_aligned)

	clrlwi	0,0,29
	mtocrf	0x01,0
	subf	5,0,5

	/* Get DST aligned to 8 bytes.  */
1:	bf	31,2f

	stb	4,0(10)
	addi	10,10,1
2:	bf	30,4f

	sth	4,0(10)
	addi	10,10,2
4:	bf	29,L(big_aligned)

	stw	4,0(10)
	addi	10,10,4

	.align	4
L(big_aligned):
	cmplwi	cr5,5,255
	li	0,32
	cmplwi	cr1,5,160
	dcbtst	0,10
	cmplwi	cr6,4,0
	srwi	9,5,3		/* Number of full doublewords remaining.  */
	crand	27,26,21
	mtocrf	0x01,9
	bt	27,L(huge)

	/* From this point on, we'll copy 32+ bytes and the value
	   isn't 0 (so we can't use dcbz).  */

	srwi	8,5,5
	clrlwi	11,5,29
	cmplwi	cr6,11,0
	cmplwi	cr1,9,4
	mtctr	8

	/* Copy 1~3 doublewords so the main loop starts
	at a multiple of 32 bytes.  */

	bf	30,1f

	stw	4,0(10)
	stw	4,4(10)
	stw	4,8(10)
	stw	4,12(10)
	addi	10,10,16
	bf	31,L(big_loop)

	stw	4,0(10)
	stw	4,4(10)
	addi	10,10,8
	mr	12,10
	blt	cr1,L(tail_bytes)

	b	L(big_loop)

	.align	4
1:	/* Copy 1 doubleword.  */
	bf	31,L(big_loop)

	stw	4,0(10)
	stw	4,4(10)
	addi	10,10,8

	/* First use a 32-bytes loop with stw's to try and avoid the LHS due
	   to the lfd we will do next.  Also, ping-pong through r10 and r12
	   to avoid AGEN delays.  */
	.align	4
L(big_loop):
	addi	12,10,32
	stw	4,0(10)
	stw	4,4(10)
	stw	4,8(10)
	stw	4,12(10)
	stw	4,16(10)
	stw	4,20(10)
	stw	4,24(10)
	stw	4,28(10)
	bdz	L(tail_bytes)

	addi	10,10,64
	stw	4,0(12)
	stw	4,4(12)
	stw	4,8(12)
	stw	4,12(12)
	stw	4,16(12)
	stw	4,20(12)
	stw	4,24(12)
	stw	4,28(12)
	bdnz	L(big_loop_fast_setup)

	mr	12,10
	b	L(tail_bytes)

	/* Now that we're probably past the LHS window, use the VSX to
	   speed up the loop.  */
L(big_loop_fast_setup):
	li	11,24
	li	6,16
	lxvdsx	4,1,11

	.align	4
L(big_loop_fast):
	addi	12,10,32
	stxvd2x	4,0,10
	stxvd2x	4,10,6
	bdz	L(tail_bytes)

	addi	10,10,64
	stxvd2x	4,0,12
	stxvd2x	4,12,6
	bdnz	L(big_loop_fast)

	mr	12,10

	.align	4
L(tail_bytes):

	/* Check for tail bytes.  */
	mr	1,7		/* Restore r1.  */
	beqlr	cr6

	clrlwi	0,5,29
	mtocrf	0x01,0

	/*  At this point we have a tail of 0-7 bytes and we know that the
	destination is doubleword-aligned.  */
4:	/* Copy 4 bytes.  */
	bf	29,2f

	stw	4,0(12)
	addi	12,12,4
2:	/* Copy 2 bytes.  */
	bf	30,1f

	sth	4,0(12)
	addi	12,12,2
1:	/* Copy 1 byte.  */
	bflr	31

	stb	4,0(12)
	blr


	/* Special case when value is 0 and we have a long length to deal
	   with.  Use dcbz to zero out 128-bytes at a time.  Before using
	   dcbz though, we need to get the destination 128-bytes aligned.  */
	.align	4
L(huge):
	lfd	4,24(1)
	andi.	11,10,127
	neg	0,10
	beq	L(huge_aligned)

	clrlwi	0,0,25
	subf	5,0,5
	srwi	0,0,3
	mtocrf  0x01,0

	/* Get DST aligned to 128 bytes.  */
8:	bf	28,4f

	stfd	4,0(10)
	stfd	4,8(10)
	stfd	4,16(10)
	stfd	4,24(10)
	stfd	4,32(10)
	stfd	4,40(10)
	stfd	4,48(10)
	stfd	4,56(10)
	addi	10,10,64
	.align	4
4:	bf	29,2f

	stfd	4,0(10)
	stfd	4,8(10)
	stfd	4,16(10)
	stfd	4,24(10)
	addi	10,10,32
	.align	4
2:	bf	30,1f

	stfd	4,0(10)
	stfd	4,8(10)
	addi	10,10,16
	.align	4
1:	bf	31,L(huge_aligned)

	stfd	4,0(10)
	addi	10,10,8

L(huge_aligned):
	srwi	8,5,7
	clrlwi	11,5,25
	cmplwi	cr6,11,0
	mtctr	8

	/* Copies 128-bytes at a time.  */
	.align	4
L(huge_loop):
	dcbz	0,10
	addi	10,10,128
	bdnz	L(huge_loop)

	/* We have a tail of 0~127 bytes to handle.  */
	mr	1,7		/* Restore r1.  */
	beqlr	cr6

	subf	9,3,10
	subf	5,9,12
	srwi	8,5,3
	cmplwi	cr6,8,0
	mtocrf	0x01,8

	/* We have a tail o 1~127 bytes. Copy up to 15 doublewords for
	speed.  We'll handle the resulting tail bytes later.  */
	beq	cr6,L(tail)

8:	bf	28,4f

	stfd	4,0(10)
	stfd	4,8(10)
	stfd	4,16(10)
	stfd	4,24(10)
	stfd	4,32(10)
	stfd	4,40(10)
	stfd	4,48(10)
	stfd	4,56(10)
	addi	10,10,64
	.align	4
4:	bf	29,2f

	stfd	4,0(10)
	stfd	4,8(10)
	stfd	4,16(10)
	stfd	4,24(10)
	addi	10,10,32
	.align	4
2:	bf	30,1f

	stfd	4,0(10)
	stfd	4,8(10)
	addi	10,10,16
	.align	4
1:	bf	31,L(tail)

	stfd	4,0(10)
	addi	10,10,8

	/* Handle the rest of the tail bytes here.  */
L(tail):
	mtocrf	0x01,5

	.align	4
4:	bf	29,2f

	stw	4,0(10)
	addi	10,10,4
	.align	4
2:	bf	30,1f

	sth	4,0(10)
	addi	10,10,2
	.align	4
1:	bflr	31

	stb	4,0(10)
	blr


	/* Expanded tree to copy tail bytes without increments.  */
	.align	4
L(copy_tail):
	bf	29,L(FXX)

	stw	4,0(10)
	bf	30,L(TFX)

	sth	4,4(10)
	bflr	31

	stb	4,6(10)
	blr

	.align	4
L(FXX):	bf	30,L(FFX)

	sth	4,0(10)
	bflr	31

	stb	4,2(10)
	blr

	.align	4
L(TFX):	bflr	31

	stb	4,4(10)
	blr

	.align	4
L(FFX):	bflr	31

	stb	4,0(10)
	blr

	/* Handle copies of 9~31 bytes.  */
	.align	4
L(medium):
	/* At least 9 bytes to go.  */
	andi.	11,10,3
	clrlwi	0,0,30
	beq	L(medium_aligned)

	/* Force 4-bytes alignment for DST.  */
	mtocrf	0x01,0
	subf	5,0,5
1:	/* Copy 1 byte.  */
	bf	31,2f

	stb	4,0(10)
	addi	10,10,1
2:	/* Copy 2 bytes.  */
	bf	30,L(medium_aligned)

	sth	4,0(10)
	addi	10,10,2

	.align	4
L(medium_aligned):
	/* At least 6 bytes to go, and DST is word-aligned.  */
	cmplwi	cr1,5,16
	mtocrf	0x01,5
	blt	cr1,8f

	/* Copy 16 bytes.  */
	stw	4,0(10)
	stw	4,4(10)
	stw	4,8(10)
	stw	4,12(10)
	addi	10,10,16
8:	/* Copy 8 bytes.  */
	bf	28,4f

	stw	4,0(10)
	stw	4,4(10)
	addi	10,10,8
4:	/* Copy 4 bytes.  */
	bf	29,2f

	stw	4,0(10)
	addi	10,10,4
2:	/* Copy 2-3 bytes.  */
	bf	30,1f

	sth	4,0(10)
	addi	10,10,2
1:	/* Copy 1 byte.  */
	bflr	31

	stb	4,0(10)
	blr

	/* Handles copies of 0~8 bytes.  */
	.align	4
L(small):
	mtocrf	0x01,5
	bne	cr6,L(copy_tail)

	stw	4,0(10)
	stw	4,4(10)
	blr

END (memset)
libc_hidden_builtin_def (memset)
