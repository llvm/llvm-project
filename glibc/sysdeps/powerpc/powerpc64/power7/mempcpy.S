/* Optimized mempcpy implementation for POWER7.
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


/* void * [r3] __mempcpy (void *dst [r3], void *src [r4], size_t len [r5]);
    Returns 'dst' + 'len'.  */

#ifndef MEMPCPY
# define MEMPCPY __mempcpy
#endif
	.machine  power7
ENTRY_TOCLESS (MEMPCPY, 5)
	CALL_MCOUNT 3

	cmpldi	cr1,5,31
	neg	0,3
	std	3,-16(1)
	std	31,-8(1)
	cfi_offset(31,-8)
	ble	cr1,L(copy_LT_32)   /* If move < 32 bytes use short move
				       code.  */

	andi.	11,3,7	      /* Check alignment of DST.  */


	clrldi	10,4,61	      /* Check alignment of SRC.  */
	cmpld	cr6,10,11     /* SRC and DST alignments match?  */
	mr	12,4
	mr	31,5
	bne	cr6,L(copy_GE_32_unaligned)

	srdi	9,5,3	      /* Number of full quadwords remaining.  */

	beq	L(copy_GE_32_aligned_cont)

	clrldi	0,0,61
	mtcrf	0x01,0
	subf	31,0,5

	/* Get the SRC aligned to 8 bytes.  */

1:	bf	31,2f
	lbz	6,0(12)
	addi	12,12,1
	stb	6,0(3)
	addi	3,3,1
2:	bf	30,4f
	lhz	6,0(12)
	addi	12,12,2
	sth	6,0(3)
	addi	3,3,2
4:	bf	29,0f
	lwz	6,0(12)
	addi	12,12,4
	stw	6,0(3)
	addi	3,3,4
0:
	clrldi	10,12,61      /* Check alignment of SRC again.  */
	srdi	9,31,3	      /* Number of full doublewords remaining.  */

L(copy_GE_32_aligned_cont):

	clrldi	11,31,61
	mtcrf	0x01,9

	srdi	8,31,5
	cmpldi	cr1,9,4
	cmpldi	cr6,11,0
	mr	11,12

	/* Copy 1~3 doublewords so the main loop starts
	at a multiple of 32 bytes.  */

	bf	30,1f
	ld	6,0(12)
	ld	7,8(12)
	addi	11,12,16
	mtctr	8
	std	6,0(3)
	std	7,8(3)
	addi	10,3,16
	bf	31,4f
	ld	0,16(12)
	std	0,16(3)
	blt	cr1,3f
	addi	11,12,24
	addi	10,3,24
	b	4f

	.align	4
1:	/* Copy 1 doubleword and set the counter.  */
	mr	10,3
	mtctr	8
	bf	31,4f
	ld	6,0(12)
	addi	11,12,8
	std	6,0(3)
	addi	10,3,8

	/* Main aligned copy loop. Copies 32-bytes at a time.  */
	.align	4
4:
	ld	6,0(11)
	ld	7,8(11)
	ld	8,16(11)
	ld	0,24(11)
	addi	11,11,32

	std	6,0(10)
	std	7,8(10)
	std	8,16(10)
	std	0,24(10)
	addi	10,10,32
	bdnz	4b
3:

	/* Check for tail bytes.  */
	rldicr	0,31,0,60
	mtcrf	0x01,31
	beq	cr6,0f

.L9:
	add	3,3,0
	add	12,12,0

	/*  At this point we have a tail of 0-7 bytes and we know that the
	destination is doubleword-aligned.  */
4:	/* Copy 4 bytes.  */
	bf	29,2f

	lwz	6,0(12)
	addi	12,12,4
	stw	6,0(3)
	addi	3,3,4
2:	/* Copy 2 bytes.  */
	bf	30,1f

	lhz	6,0(12)
	addi	12,12,2
	sth	6,0(3)
	addi	3,3,2
1:	/* Copy 1 byte.  */
	bf	31,0f

	lbz	6,0(12)
	stb	6,0(3)
0:	/* Return DST + LEN pointer.  */
	ld	31,-8(1)
	ld	3,-16(1)
	add	3,3,5
	blr

	/* Handle copies of 0~31 bytes.  */
	.align	4
L(copy_LT_32):
	cmpldi	cr6,5,8
	mr	12,4
	mtcrf	0x01,5
	ble	cr6,L(copy_LE_8)

	/* At least 9 bytes to go.  */
	neg	8,4
	clrrdi	11,4,2
	andi.	0,8,3
	cmpldi	cr1,5,16
	mr	10,5
	beq	L(copy_LT_32_aligned)

	/* Force 4-bytes alignment for SRC.  */
	mtocrf  0x01,0
	subf	10,0,5
2:	bf	30,1f

	lhz	6,0(12)
	addi	12,12,2
	sth	6,0(3)
	addi	3,3,2
1:	bf	31,L(end_4bytes_alignment)

	lbz	6,0(12)
	addi	12,12,1
	stb	6,0(3)
	addi	3,3,1

	.align	4
L(end_4bytes_alignment):
	cmpldi	cr1,10,16
	mtcrf	0x01,10

L(copy_LT_32_aligned):
	/* At least 6 bytes to go, and SRC is word-aligned.  */
	blt	cr1,8f

	/* Copy 16 bytes.  */
	lwz	6,0(12)
	lwz	7,4(12)
	stw	6,0(3)
	lwz	8,8(12)
	stw	7,4(3)
	lwz	6,12(12)
	addi	12,12,16
	stw	8,8(3)
	stw	6,12(3)
	addi	3,3,16
8:	/* Copy 8 bytes.  */
	bf	28,4f

	lwz	6,0(12)
	lwz	7,4(12)
	addi	12,12,8
	stw	6,0(3)
	stw	7,4(3)
	addi	3,3,8
4:	/* Copy 4 bytes.  */
	bf	29,2f

	lwz	6,0(12)
	addi	12,12,4
	stw	6,0(3)
	addi	3,3,4
2:	/* Copy 2-3 bytes.  */
	bf	30,1f

	lhz	6,0(12)
	sth	6,0(3)
	bf	31,0f
	lbz	7,2(12)
	stb	7,2(3)
	ld	3,-16(1)
	add	3,3,5
	blr

	.align	4
1:	/* Copy 1 byte.  */
	bf	31,0f

	lbz	6,0(12)
	stb	6,0(3)
0:	/* Return DST + LEN pointer.  */
	ld	3,-16(1)
	add	3,3,5
	blr

	/* Handles copies of 0~8 bytes.  */
	.align	4
L(copy_LE_8):
	bne	cr6,4f

	/* Though we could've used ld/std here, they are still
	slow for unaligned cases.  */

	lwz	6,0(4)
	lwz	7,4(4)
	stw	6,0(3)
	stw	7,4(3)
	ld	3,-16(1)      /* Return DST + LEN pointer.  */
	add	3,3,5
	blr

	.align	4
4:	/* Copies 4~7 bytes.  */
	bf	29,2b

	lwz	6,0(4)
	stw	6,0(3)
	bf	30,5f
	lhz	7,4(4)
	sth	7,4(3)
	bf	31,0f
	lbz	8,6(4)
	stb	8,6(3)
	ld	3,-16(1)
	add	3,3,5
	blr

	.align	4
5:	/* Copy 1 byte.  */
	bf	31,0f

	lbz	6,4(4)
	stb	6,4(3)

0:	/* Return DST + LEN pointer.  */
	ld	3,-16(1)
	add	3,3,5
	blr

	/* Handle copies of 32+ bytes where DST is aligned (to quadword) but
	SRC is not.  Use aligned quadword loads from SRC, shifted to realign
	the data, allowing for aligned DST stores.  */
	.align	4
L(copy_GE_32_unaligned):
	clrldi	0,0,60	      /* Number of bytes until the 1st
				 quadword.  */
	andi.	11,3,15	      /* Check alignment of DST (against
				 quadwords).  */
	srdi	9,5,4	      /* Number of full quadwords remaining.  */

	beq	L(copy_GE_32_unaligned_cont)

	/* SRC is not quadword aligned, get it aligned.  */

	mtcrf	0x01,0
	subf	31,0,5

	/* Vector instructions work best when proper alignment (16-bytes)
	is present.  Move 0~15 bytes as needed to get DST quadword-aligned.  */
1:	/* Copy 1 byte.  */
	bf	31,2f

	lbz	6,0(12)
	addi	12,12,1
	stb	6,0(3)
	addi	3,3,1
2:	/* Copy 2 bytes.  */
	bf	30,4f

	lhz	6,0(12)
	addi	12,12,2
	sth	6,0(3)
	addi	3,3,2
4:	/* Copy 4 bytes.  */
	bf	29,8f

	lwz	6,0(12)
	addi	12,12,4
	stw	6,0(3)
	addi	3,3,4
8:	/* Copy 8 bytes.  */
	bf	28,0f

	ld	6,0(12)
	addi	12,12,8
	std	6,0(3)
	addi	3,3,8
0:
	clrldi	10,12,60      /* Check alignment of SRC.  */
	srdi	9,31,4	      /* Number of full quadwords remaining.  */

	/* The proper alignment is present, it is OK to copy the bytes now.  */
L(copy_GE_32_unaligned_cont):

	/* Setup two indexes to speed up the indexed vector operations.  */
	clrldi	11,31,60
	li	6,16	      /* Index for 16-bytes offsets.  */
	li	7,32	      /* Index for 32-bytes offsets.  */
	cmpldi	cr1,11,0
	srdi	8,31,5	      /* Setup the loop counter.  */
	mr	10,3
	mr	11,12
	mtcrf	0x01,9
	cmpldi	cr6,9,1
#ifdef __LITTLE_ENDIAN__
	lvsr    5,0,12
#else
	lvsl    5,0,12
#endif
	lvx	3,0,12
	bf	31,L(setup_unaligned_loop)

	/* Copy another 16 bytes to align to 32-bytes due to the loop .  */
	lvx	4,12,6
#ifdef __LITTLE_ENDIAN__
	vperm   6,4,3,5
#else
	vperm   6,3,4,5
#endif
	addi	11,12,16
	addi	10,3,16
	stvx	6,0,3
	vor	3,4,4

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

	lvx	4,11,6	      /* vr4 = r11+16.  */
#ifdef __LITTLE_ENDIAN__
	vperm   6,4,3,5
#else
	vperm   6,3,4,5
#endif
	lvx	3,11,7	      /* vr3 = r11+32.  */
#ifdef __LITTLE_ENDIAN__
	vperm   10,3,4,5
#else
	vperm   10,4,3,5
#endif
	addi	11,11,32
	stvx	6,0,10
	stvx	10,10,6
	addi	10,10,32

	bdnz	L(unaligned_loop)

	.align	4
L(end_unaligned_loop):

	/* Check for tail bytes.  */
	rldicr	0,31,0,59
	mtcrf	0x01,31
	beq	cr1,0f

	add	3,3,0
	add	12,12,0

	/*  We have 1~15 tail bytes to copy, and DST is quadword aligned.  */
8:	/* Copy 8 bytes.  */
	bf	28,4f

	lwz	6,0(12)
	lwz	7,4(12)
	addi	12,12,8
	stw	6,0(3)
	stw	7,4(3)
	addi	3,3,8
4:	/* Copy 4 bytes.  */
	bf	29,2f

	lwz	6,0(12)
	addi	12,12,4
	stw	6,0(3)
	addi	3,3,4
2:	/* Copy 2~3 bytes.  */
	bf	30,1f

	lhz	6,0(12)
	addi	12,12,2
	sth	6,0(3)
	addi	3,3,2
1:	/* Copy 1 byte.  */
	bf	31,0f

	lbz	6,0(12)
	stb	6,0(3)
0:	/* Return DST + LEN pointer.  */
	ld	31,-8(1)
	ld	3,-16(1)
	add	3,3,5
	blr

END_GEN_TB (MEMPCPY,TB_TOCLESS)
libc_hidden_def (__mempcpy)
weak_alias (__mempcpy, mempcpy)
libc_hidden_builtin_def (mempcpy)
