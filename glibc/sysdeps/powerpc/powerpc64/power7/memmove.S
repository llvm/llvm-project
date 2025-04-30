/* Optimized memmove implementation for PowerPC64/POWER7.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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


/* void* [r3] memmove (void *dest [r3], const void *src [r4], size_t len [r5])

   This optimization check if memory 'dest'  overlaps with 'src'. If it does
   not then it calls an optimized memcpy call (similar to memcpy for POWER7,
   embedded here to gain some cycles).
   If source and destiny overlaps, a optimized backwards memcpy is used
   instead.  */

#ifndef MEMMOVE
# define MEMMOVE memmove
#endif
	.machine power7
ENTRY_TOCLESS (MEMMOVE, 5)
	CALL_MCOUNT 3

L(_memmove):
	subf    r9,r4,r3
	cmpld   cr7,r9,r5
	blt	cr7,L(memmove_bwd)

	cmpldi	cr1,r5,31
	neg	0,3
	ble	cr1, L(copy_LT_32)  /* If move < 32 bytes use short move
				       code.  */

	andi.	10,3,15
	clrldi	11,4,60
	cmpld	cr6,10,11	/* SRC and DST alignments match?  */

	mr	r11,3
	bne	cr6,L(copy_GE_32_unaligned)
	beq	L(aligned_copy)

	mtocrf	0x01,0
	clrldi	0,0,60

/* Get the DST and SRC aligned to 8 bytes (16 for little-endian).  */
1:
	bf	31,2f
	lbz	6,0(r4)
	addi	r4,r4,1
	stb	6,0(r11)
	addi	r11,r11,1
2:
	bf	30,4f
	lhz	6,0(r4)
	addi	r4,r4,2
	sth	6,0(r11)
	addi	r11,r11,2
4:
	bf	29,8f
	lwz	6,0(r4)
	addi	r4,r4,4
	stw	6,0(r11)
	addi	r11,r11,4
8:
	bf	28,16f
	ld	6,0(r4)
	addi	r4,r4,8
	std	6,0(r11)
	addi	r11,r11,8
16:
	subf	r5,0,r5

/* Main aligned copy loop. Copies 128 bytes at a time. */
L(aligned_copy):
	li	6,16
	li	7,32
	li	8,48
	mtocrf	0x02,r5
	srdi	12,r5,7
	cmpdi	12,0
	beq	L(aligned_tail)
	lvx	6,0,r4
	lvx	7,r4,6
	mtctr	12
	b	L(aligned_128loop)

	.align  4
L(aligned_128head):
	/* for the 2nd + iteration of this loop. */
	lvx	6,0,r4
	lvx	7,r4,6
L(aligned_128loop):
	lvx	8,r4,7
	lvx	9,r4,8
	stvx	6,0,r11
	addi	r4,r4,64
	stvx	7,r11,6
	stvx	8,r11,7
	stvx	9,r11,8
	lvx	6,0,r4
	lvx	7,r4,6
	addi	r11,r11,64
	lvx	8,r4,7
	lvx	9,r4,8
	addi	r4,r4,64
	stvx	6,0,r11
	stvx	7,r11,6
	stvx	8,r11,7
	stvx	9,r11,8
	addi	r11,r11,64
	bdnz	L(aligned_128head)

L(aligned_tail):
	mtocrf	0x01,r5
	bf	25,32f
	lvx	6,0,r4
	lvx	7,r4,6
	lvx	8,r4,7
	lvx	9,r4,8
	addi	r4,r4,64
	stvx	6,0,r11
	stvx	7,r11,6
	stvx	8,r11,7
	stvx	9,r11,8
	addi	r11,r11,64
32:
	bf	26,16f
	lvx	6,0,r4
	lvx	7,r4,6
	addi	r4,r4,32
	stvx	6,0,r11
	stvx	7,r11,6
	addi	r11,r11,32
16:
	bf	27,8f
	lvx	6,0,r4
	addi	r4,r4,16
	stvx	6,0,r11
	addi	r11,r11,16
8:
	bf	28,4f
	ld	6,0(r4)
	addi	r4,r4,8
	std     6,0(r11)
	addi	r11,r11,8
4:	/* Copies 4~7 bytes.  */
	bf	29,L(tail2)
	lwz	6,0(r4)
	stw     6,0(r11)
	bf      30,L(tail5)
	lhz     7,4(r4)
	sth     7,4(r11)
	bflr	31
	lbz     8,6(r4)
	stb     8,6(r11)
	/* Return original DST pointer.  */
	blr

/* Handle copies of 0~31 bytes.  */
	.align	4
L(copy_LT_32):
	mr	r11,3
	cmpldi	cr6,r5,8
	mtocrf	0x01,r5
	ble	cr6,L(copy_LE_8)

	/* At least 9 bytes to go.  */
	neg	8,4
	andi.	0,8,3
	cmpldi	cr1,r5,16
	beq	L(copy_LT_32_aligned)

	/* Force 4-byte alignment for SRC.  */
	mtocrf	0x01,0
	subf	r5,0,r5
2:
	bf	30,1f
	lhz	6,0(r4)
	addi	r4,r4,2
	sth	6,0(r11)
	addi	r11,r11,2
1:
	bf	31,L(end_4bytes_alignment)
	lbz	6,0(r4)
	addi	r4,r4,1
	stb	6,0(r11)
	addi	r11,r11,1

	.align	4
L(end_4bytes_alignment):
	cmpldi	cr1,r5,16
	mtocrf	0x01,r5

L(copy_LT_32_aligned):
	/* At least 6 bytes to go, and SRC is word-aligned.  */
	blt	cr1,8f

	/* Copy 16 bytes.  */
	lwz	6,0(r4)
	lwz	7,4(r4)
	stw	6,0(r11)
	lwz	8,8(r4)
	stw	7,4(r11)
	lwz	6,12(r4)
	addi	r4,r4,16
	stw	8,8(r11)
	stw	6,12(r11)
	addi	r11,r11,16
8:	/* Copy 8 bytes.  */
	bf	28,L(tail4)
	lwz	6,0(r4)
	lwz	7,4(r4)
	addi	r4,r4,8
	stw	6,0(r11)
	stw	7,4(r11)
	addi	r11,r11,8

	.align	4
/* Copies 4~7 bytes.  */
L(tail4):
	bf	29,L(tail2)
	lwz	6,0(r4)
	stw	6,0(r11)
	bf	30,L(tail5)
	lhz	7,4(r4)
	sth	7,4(r11)
	bflr	31
	lbz	8,6(r4)
	stb	8,6(r11)
	/* Return original DST pointer.  */
	blr

	.align	4
/* Copies 2~3 bytes.  */
L(tail2):
	bf	30,1f
	lhz	6,0(r4)
	sth	6,0(r11)
	bflr	31
	lbz	7,2(r4)
	stb	7,2(r11)
	blr

	.align	4
L(tail5):
	bflr	31
	lbz	6,4(r4)
	stb	6,4(r11)
	blr

	.align	4
1:
	bflr	31
	lbz	6,0(r4)
	stb	6,0(r11)
	/* Return original DST pointer.  */
	blr

/* Handles copies of 0~8 bytes.  */
	.align	4
L(copy_LE_8):
	bne	cr6,L(tail4)

	/* Though we could've used ld/std here, they are still
	slow for unaligned cases.  */

	lwz	6,0(r4)
	lwz	7,4(r4)
	stw	6,0(r11)
	stw	7,4(r11)
	blr


/* Handle copies of 32+ bytes where DST is aligned (to quadword) but
   SRC is not.	Use aligned quadword loads from SRC, shifted to realign
   the data, allowing for aligned DST stores.  */
	.align	4
L(copy_GE_32_unaligned):
	clrldi	0,0,60	      /* Number of bytes until the 1st r11 quadword.  */
	srdi	9,r5,4	      /* Number of full quadwords remaining.  */

	beq	L(copy_GE_32_unaligned_cont)

	/* DST is not quadword aligned, get it aligned.  */

	mtocrf	0x01,0
	subf	r5,0,r5

	/* Vector instructions work best when proper alignment (16-bytes)
	is present.  Move 0~15 bytes as needed to get DST quadword-aligned.  */
1:
	bf	31,2f
	lbz	6,0(r4)
	addi	r4,r4,1
	stb	6,0(r11)
	addi	r11,r11,1
2:
	bf	30,4f
	lhz	6,0(r4)
	addi	r4,r4,2
	sth	6,0(r11)
	addi	r11,r11,2
4:
	bf	29,8f
	lwz	6,0(r4)
	addi	r4,r4,4
	stw	6,0(r11)
	addi	r11,r11,4
8:
	bf	28,0f
	ld	6,0(r4)
	addi	r4,r4,8
	std	6,0(r11)
	addi	r11,r11,8
0:
	srdi	9,r5,4	      /* Number of full quadwords remaining.  */

	/* The proper alignment is present, it is OK to copy the bytes now.  */
L(copy_GE_32_unaligned_cont):

	/* Setup two indexes to speed up the indexed vector operations.  */
	clrldi	10,r5,60
	li	6,16	      /* Index for 16-bytes offsets.  */
	li	7,32	      /* Index for 32-bytes offsets.  */
	cmpldi	cr1,10,0
	srdi	8,r5,5	      /* Setup the loop counter.  */
	mtocrf	0x01,9
	cmpldi	cr6,9,1
#ifdef __LITTLE_ENDIAN__
	lvsr	5,0,r4
#else
	lvsl	5,0,r4
#endif
	lvx	3,0,r4
	li	0,0
	bf	31,L(setup_unaligned_loop)

	/* Copy another 16 bytes to align to 32-bytes due to the loop.  */
	lvx	4,r4,6
#ifdef __LITTLE_ENDIAN__
	vperm	6,4,3,5
#else
	vperm	6,3,4,5
#endif
	addi	r4,r4,16
	stvx	6,0,r11
	addi	r11,r11,16
	vor	3,4,4
	clrrdi	0,r4,60

L(setup_unaligned_loop):
	mtctr	8
	ble	cr6,L(end_unaligned_loop)

	/* Copy 32 bytes at a time using vector instructions.  */
	.align	4
L(unaligned_loop):

	/* Note: vr6/vr10 may contain data that was already copied,
	but in order to get proper alignment, we may have to copy
	some portions again. This is faster than having unaligned
	vector instructions though.  */

	lvx	4,r4,6
#ifdef __LITTLE_ENDIAN__
	vperm	6,4,3,5
#else
	vperm	6,3,4,5
#endif
	lvx	3,r4,7
#ifdef __LITTLE_ENDIAN__
	vperm	10,3,4,5
#else
	vperm	10,4,3,5
#endif
	addi	r4,r4,32
	stvx	6,0,r11
	stvx	10,r11,6
	addi	r11,r11,32
	bdnz	L(unaligned_loop)

	clrrdi	0,r4,60

	.align	4
L(end_unaligned_loop):

	/* Check for tail bytes.  */
	mtocrf	0x01,r5
	beqlr	cr1

	add	r4,r4,0

	/*  We have 1~15 tail bytes to copy, and DST is quadword aligned.  */
	/* Copy 8 bytes.  */
	bf	28,4f
	lwz	6,0(r4)
	lwz	7,4(r4)
	addi	r4,r4,8
	stw	6,0(r11)
	stw	7,4(r11)
	addi	r11,r11,8
4:	/* Copy 4~7 bytes.  */
	bf	29,L(tail2)
	lwz	6,0(r4)
	stw	6,0(r11)
	bf	30,L(tail5)
	lhz	7,4(r4)
	sth	7,4(r11)
	bflr	31
	lbz	8,6(r4)
	stb	8,6(r11)
	/* Return original DST pointer.  */
	blr

	/* Start to memcpy backward implementation: the algorith first check if
	   src and dest have the same alignment and if it does align both to 16
	   bytes and copy using VSX instructions.
	   If does not, align dest to 16 bytes and use VMX (altivec) instruction
	   to read two 16 bytes at time, shift/permute the bytes read and write
	   aligned to dest.  */
L(memmove_bwd):
	cmpldi	cr1,r5,31
	/* Copy is done backwards: update the pointers and check alignment.  */
	add	r11,r3,r5
	add	r4,r4,r5
	mr	r0,r11
	ble	cr1, L(copy_LT_32_bwd)  /* If move < 32 bytes use short move
				           code.  */

	andi.	r10,r11,15	    /* Check if r11 is aligned to 16 bytes  */
	clrldi	r9,r4,60	    /* Check if r4 is aligned to 16 bytes  */
	cmpld	cr6,r10,r9	    /* SRC and DST alignments match?  */

	bne     cr6,L(copy_GE_32_unaligned_bwd)
	beq     L(aligned_copy_bwd)

	mtocrf	0x01,r0
	clrldi	r0,r0,60

/* Get the DST and SRC aligned to 16 bytes.  */
1:
	bf	31,2f
	lbz	r6,-1(r4)
	subi	r4,r4,1
	stb	r6,-1(r11)
	subi	r11,r11,1
2:
	bf	30,4f
	lhz	r6,-2(r4)
	subi	r4,r4,2
	sth	r6,-2(r11)
	subi	r11,r11,2
4:
	bf	29,8f
	lwz	r6,-4(r4)
	subi	r4,r4,4
	stw	r6,-4(r11)
	subi	r11,r11,4
8:
	bf	28,16f
	ld	r6,-8(r4)
	subi	r4,r4,8
	std	r6,-8(r11)
	subi	r11,r11,8
16:
	subf	r5,0,r5

/* Main aligned copy loop. Copies 128 bytes at a time. */
L(aligned_copy_bwd):
	li	r6,-16
	li	r7,-32
	li	r8,-48
	li	r9,-64
	mtocrf	0x02,r5
	srdi	r12,r5,7
	cmpdi	r12,0
	beq	L(aligned_tail_bwd)
	lvx	v6,r4,r6
	lvx	v7,r4,r7
	mtctr	12
	b	L(aligned_128loop_bwd)

	.align  4
L(aligned_128head_bwd):
	/* for the 2nd + iteration of this loop. */
	lvx	v6,r4,r6
	lvx	v7,r4,r7
L(aligned_128loop_bwd):
	lvx	v8,r4,r8
	lvx	v9,r4,r9
	stvx	v6,r11,r6
	subi	r4,r4,64
	stvx	v7,r11,r7
	stvx	v8,r11,r8
	stvx	v9,r11,r9
	lvx	v6,r4,r6
	lvx	v7,r4,7
	subi	r11,r11,64
	lvx	v8,r4,r8
	lvx	v9,r4,r9
	subi	r4,r4,64
	stvx	v6,r11,r6
	stvx	v7,r11,r7
	stvx	v8,r11,r8
	stvx	v9,r11,r9
	subi	r11,r11,64
	bdnz	L(aligned_128head_bwd)

L(aligned_tail_bwd):
	mtocrf	0x01,r5
	bf	25,32f
	lvx	v6,r4,r6
	lvx	v7,r4,r7
	lvx	v8,r4,r8
	lvx	v9,r4,r9
	subi	r4,r4,64
	stvx	v6,r11,r6
	stvx	v7,r11,r7
	stvx	v8,r11,r8
	stvx	v9,r11,r9
	subi	r11,r11,64
32:
	bf	26,16f
	lvx	v6,r4,r6
	lvx	v7,r4,r7
	subi	r4,r4,32
	stvx	v6,r11,r6
	stvx	v7,r11,r7
	subi	r11,r11,32
16:
	bf	27,8f
	lvx	v6,r4,r6
	subi	r4,r4,16
	stvx	v6,r11,r6
	subi	r11,r11,16
8:
	bf	28,4f
	ld	r6,-8(r4)
	subi	r4,r4,8
	std     r6,-8(r11)
	subi	r11,r11,8
4:	/* Copies 4~7 bytes.  */
	bf	29,L(tail2_bwd)
	lwz	r6,-4(r4)
	stw     r6,-4(r11)
	bf      30,L(tail5_bwd)
	lhz     r7,-6(r4)
	sth     r7,-6(r11)
	bflr	31
	lbz     r8,-7(r4)
	stb     r8,-7(r11)
	/* Return original DST pointer.  */
	blr

/* Handle copies of 0~31 bytes.  */
	.align	4
L(copy_LT_32_bwd):
	cmpldi	cr6,r5,8
	mtocrf	0x01,r5
	ble	cr6,L(copy_LE_8_bwd)

	/* At least 9 bytes to go.  */
	neg	r8,r4
	andi.	r0,r8,3
	cmpldi	cr1,r5,16
	beq	L(copy_LT_32_aligned_bwd)

	/* Force 4-byte alignment for SRC.  */
	mtocrf	0x01,0
	subf	r5,0,r5
2:
	bf	30,1f
	lhz	r6,-2(r4)
	subi	r4,r4,2
	sth	r6,-2(r11)
	subi	r11,r11,2
1:
	bf	31,L(end_4bytes_alignment_bwd)
	lbz	6,-1(r4)
	subi	r4,r4,1
	stb	6,-1(r11)
	subi	r11,r11,1

	.align	4
L(end_4bytes_alignment_bwd):
	cmpldi	cr1,r5,16
	mtocrf	0x01,r5

L(copy_LT_32_aligned_bwd):
	/* At least 6 bytes to go, and SRC is word-aligned.  */
	blt	cr1,8f

	/* Copy 16 bytes.  */
	lwz	r6,-4(r4)
	lwz	r7,-8(r4)
	stw	r6,-4(r11)
	lwz	r8,-12(r4)
	stw	r7,-8(r11)
	lwz	r6,-16(r4)
	subi	r4,r4,16
	stw	r8,-12(r11)
	stw	r6,-16(r11)
	subi	r11,r11,16
8:	/* Copy 8 bytes.  */
	bf	28,L(tail4_bwd)
	lwz	r6,-4(r4)
	lwz	r7,-8(r4)
	subi	r4,r4,8
	stw	r6,-4(r11)
	stw	r7,-8(r11)
	subi	r11,r11,8

	.align	4
/* Copies 4~7 bytes.  */
L(tail4_bwd):
	bf	29,L(tail2_bwd)
	lwz	6,-4(r4)
	stw	6,-4(r11)
	bf	30,L(tail5_bwd)
	lhz	7,-6(r4)
	sth	7,-6(r11)
	bflr	31
	lbz	8,-7(r4)
	stb	8,-7(r11)
	/* Return original DST pointer.  */
	blr

	.align	4
/* Copies 2~3 bytes.  */
L(tail2_bwd):
	bf	30,1f
	lhz	6,-2(r4)
	sth	6,-2(r11)
	bflr	31
	lbz	7,-3(r4)
	stb	7,-3(r11)
	blr

	.align	4
L(tail5_bwd):
	bflr	31
	lbz	6,-5(r4)
	stb	6,-5(r11)
	blr

	.align	4
1:
	bflr	31
	lbz	6,-1(r4)
	stb	6,-1(r11)
	/* Return original DST pointer.  */
	blr


/* Handles copies of 0~8 bytes.  */
	.align	4
L(copy_LE_8_bwd):
	bne	cr6,L(tail4_bwd)

	/* Though we could've used ld/std here, they are still
	   slow for unaligned cases.  */
	lwz	6,-8(r4)
	lwz	7,-4(r4)
	stw	6,-8(r11)
	stw	7,-4(r11)
	blr


/* Handle copies of 32+ bytes where DST is aligned (to quadword) but
   SRC is not.	Use aligned quadword loads from SRC, shifted to realign
   the data, allowing for aligned DST stores.  */
	.align	4
L(copy_GE_32_unaligned_bwd):
	andi.	r10,r11,15      /* Check alignment of DST against 16 bytes..  */
	srdi	r9,r5,4		/* Number of full quadwords remaining.  */

	beq	L(copy_GE_32_unaligned_cont_bwd)

	/* DST is not quadword aligned and r10 holds the address masked to
           compare alignments.  */
	mtocrf	0x01,r10
	subf	r5,r10,r5

	/* Vector instructions work best when proper alignment (16-bytes)
	is present.  Move 0~15 bytes as needed to get DST quadword-aligned.  */
1:
	bf	31,2f
	lbz	r6,-1(r4)
	subi	r4,r4,1
	stb	r6,-1(r11)
	subi	r11,r11,1
2:
	bf	30,4f
	lhz	r6,-2(r4)
	subi	r4,r4,2
	sth	r6,-2(r11)
	subi	r11,r11,2
4:
	bf	29,8f
	lwz	r6,-4(r4)
	subi	r4,r4,4
	stw	r6,-4(r11)
	subi	r11,r11,4
8:
	bf	28,0f
	ld	r6,-8(r4)
	subi	r4,r4,8
	std	r6,-8(r11)
	subi	r11,r11,8
0:
	srdi	r9,r5,4	      /* Number of full quadwords remaining.  */

	/* The proper alignment is present, it is OK to copy the bytes now.  */
L(copy_GE_32_unaligned_cont_bwd):

	/* Setup two indexes to speed up the indexed vector operations.  */
	clrldi	r10,r5,60
	li	r6,-16	      /* Index for 16-bytes offsets.  */
	li	r7,-32	      /* Index for 32-bytes offsets.  */
	cmpldi	cr1,10,0
	srdi	r8,r5,5	      /* Setup the loop counter.  */
	mtocrf	0x01,9
	cmpldi	cr6,r9,1
#ifdef __LITTLE_ENDIAN__
	lvsr	v5,r0,r4
#else
	lvsl	v5,r0,r4
#endif
	lvx	v3,0,r4
	li	r0,0
	bf	31,L(setup_unaligned_loop_bwd)

	/* Copy another 16 bytes to align to 32-bytes due to the loop.  */
	lvx	v4,r4,r6
#ifdef __LITTLE_ENDIAN__
	vperm	v6,v3,v4,v5
#else
	vperm	v6,v4,v3,v5
#endif
	subi	r4,r4,16
	stvx	v6,r11,r6
	subi	r11,r11,16
	vor	v3,v4,v4
	clrrdi	r0,r4,60

L(setup_unaligned_loop_bwd):
	mtctr	r8
	ble	cr6,L(end_unaligned_loop_bwd)

	/* Copy 32 bytes at a time using vector instructions.  */
	.align	4
L(unaligned_loop_bwd):

	/* Note: vr6/vr10 may contain data that was already copied,
	but in order to get proper alignment, we may have to copy
	some portions again. This is faster than having unaligned
	vector instructions though.  */

	lvx	v4,r4,r6
#ifdef __LITTLE_ENDIAN__
	vperm	v6,v3,v4,v5
#else
	vperm	v6,v4,v3,v5
#endif
	lvx	v3,r4,r7
#ifdef __LITTLE_ENDIAN__
	vperm	v10,v4,v3,v5
#else
	vperm	v10,v3,v4,v5
#endif
	subi	r4,r4,32
	stvx	v6,r11,r6
	stvx	v10,r11,r7
	subi	r11,r11,32
	bdnz	L(unaligned_loop_bwd)

	clrrdi	r0,r4,60

	.align	4
L(end_unaligned_loop_bwd):

	/* Check for tail bytes.  */
	mtocrf	0x01,r5
	beqlr	cr1

	add	r4,r4,0

	/*  We have 1~15 tail bytes to copy, and DST is quadword aligned.  */
	/* Copy 8 bytes.  */
	bf	28,4f
	lwz	r6,-4(r4)
	lwz	r7,-8(r4)
	subi	r4,r4,8
	stw	r6,-4(r11)
	stw	r7,-8(r11)
	subi	r11,r11,8
4:	/* Copy 4~7 bytes.  */
	bf	29,L(tail2_bwd)
	lwz	r6,-4(r4)
	stw	r6,-4(r11)
	bf	30,L(tail5_bwd)
	lhz	r7,-6(r4)
	sth	r7,-6(r11)
	bflr	31
	lbz	r8,-7(r4)
	stb	r8,-7(r11)
	/* Return original DST pointer.  */
	blr
END_GEN_TB (MEMMOVE, TB_TOCLESS)
libc_hidden_builtin_def (memmove)


/* void bcopy(const void *src [r3], void *dest [r4], size_t n [r5])
   Implemented in this file to avoid linker create a stub function call
   in the branch to '_memmove'.  */
ENTRY_TOCLESS (__bcopy)
	mr	r6,r3
	mr	r3,r4
	mr	r4,r6
	b	L(_memmove)
END (__bcopy)
#ifndef __bcopy
weak_alias (__bcopy, bcopy)
#endif
