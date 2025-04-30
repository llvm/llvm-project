/* Optimized memcpy implementation for PowerPC64/POWER7.
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


/* void * [r3] memcpy (void *dst [r3], void *src [r4], size_t len [r5]);
   Returns 'dst'.  */

#ifndef MEMCPY
# define MEMCPY memcpy
#endif

#define dst 11		/* Use r11 so r3 kept unchanged.  */
#define src 4
#define cnt 5

	.machine power7
ENTRY_TOCLESS (MEMCPY, 5)
	CALL_MCOUNT 3

	cmpldi	cr1,cnt,31
	neg	0,3
	ble	cr1, L(copy_LT_32)  /* If move < 32 bytes use short move
				    code.  */

/* Align copies using VSX instructions to quadword. It is to avoid alignment
   traps when memcpy is used on non-cacheable memory (for instance, memory
   mapped I/O).  */
	andi.	10,3,15
	clrldi	11,4,60
	cmpld	cr6,10,11	/* SRC and DST alignments match?  */

	mr	dst,3
	bne	cr6,L(copy_GE_32_unaligned)
	beq	L(aligned_copy)

	mtocrf	0x01,0
	clrldi	0,0,60

/* Get the DST and SRC aligned to 16 bytes.  */
1:
	bf	31,2f
	lbz	6,0(src)
	addi	src,src,1
	stb	6,0(dst)
	addi	dst,dst,1
2:
	bf	30,4f
	lhz	6,0(src)
	addi	src,src,2
	sth	6,0(dst)
	addi	dst,dst,2
4:
	bf	29,8f
	lwz	6,0(src)
	addi	src,src,4
	stw	6,0(dst)
	addi	dst,dst,4
8:
	bf	28,16f
	ld	6,0(src)
	addi	src,src,8
	std	6,0(dst)
	addi	dst,dst,8
16:
	subf	cnt,0,cnt

/* Main aligned copy loop. Copies 128 bytes at a time. */
L(aligned_copy):
	li	6,16
	li	7,32
	li	8,48
	mtocrf	0x02,cnt
	srdi	12,cnt,7
	cmpdi	12,0
	beq	L(aligned_tail)
	lvx	6,0,src
	lvx	7,src,6
	mtctr	12
	b	L(aligned_128loop)

	.align  4
L(aligned_128head):
	/* for the 2nd + iteration of this loop. */
	lvx	6,0,src
	lvx	7,src,6
L(aligned_128loop):
	lvx	8,src,7
	lvx	9,src,8
	stvx	6,0,dst
	addi	src,src,64
	stvx	7,dst,6
	stvx	8,dst,7
	stvx	9,dst,8
	lvx	6,0,src
	lvx	7,src,6
	addi	dst,dst,64
	lvx	8,src,7
	lvx	9,src,8
	addi	src,src,64
	stvx	6,0,dst
	stvx	7,dst,6
	stvx	8,dst,7
	stvx	9,dst,8
	addi	dst,dst,64
	bdnz	L(aligned_128head)

L(aligned_tail):
	mtocrf	0x01,cnt
	bf	25,32f
	lvx	6,0,src
	lvx	7,src,6
	lvx	8,src,7
	lvx	9,src,8
	addi	src,src,64
	stvx	6,0,dst
	stvx	7,dst,6
	stvx	8,dst,7
	stvx	9,dst,8
	addi	dst,dst,64
32:
	bf	26,16f
	lvx	6,0,src
	lvx	7,src,6
	addi	src,src,32
	stvx	6,0,dst
	stvx	7,dst,6
	addi	dst,dst,32
16:
	bf	27,8f
	lvx	6,0,src
	addi	src,src,16
	stvx	6,0,dst
	addi	dst,dst,16
8:
	bf	28,4f
	ld	6,0(src)
	addi	src,src,8
	std     6,0(dst)
	addi	dst,dst,8
4:	/* Copies 4~7 bytes.  */
	bf	29,L(tail2)
	lwz	6,0(src)
	stw     6,0(dst)
	bf      30,L(tail5)
	lhz     7,4(src)
	sth     7,4(dst)
	bflr	31
	lbz     8,6(src)
	stb     8,6(dst)
	/* Return original DST pointer.  */
	blr


/* Handle copies of 0~31 bytes.  */
	.align	4
L(copy_LT_32):
	mr	dst,3
	cmpldi	cr6,cnt,8
	mtocrf	0x01,cnt
	ble	cr6,L(copy_LE_8)

	/* At least 9 bytes to go.  */
	neg	8,4
	andi.	0,8,3
	cmpldi	cr1,cnt,16
	beq	L(copy_LT_32_aligned)

	/* Force 4-byte alignment for SRC.  */
	mtocrf	0x01,0
	subf	cnt,0,cnt
2:
	bf	30,1f
	lhz	6,0(src)
	addi	src,src,2
	sth	6,0(dst)
	addi	dst,dst,2
1:
	bf	31,L(end_4bytes_alignment)
	lbz	6,0(src)
	addi	src,src,1
	stb	6,0(dst)
	addi	dst,dst,1

	.align	4
L(end_4bytes_alignment):
	cmpldi	cr1,cnt,16
	mtocrf	0x01,cnt

L(copy_LT_32_aligned):
	/* At least 6 bytes to go, and SRC is word-aligned.  */
	blt	cr1,8f

	/* Copy 16 bytes.  */
	lwz	6,0(src)
	lwz	7,4(src)
	stw	6,0(dst)
	lwz	8,8(src)
	stw	7,4(dst)
	lwz	6,12(src)
	addi	src,src,16
	stw	8,8(dst)
	stw	6,12(dst)
	addi	dst,dst,16
8:	/* Copy 8 bytes.  */
	bf	28,L(tail4)
	lwz	6,0(src)
	lwz	7,4(src)
	addi	src,src,8
	stw	6,0(dst)
	stw	7,4(dst)
	addi	dst,dst,8

	.align	4
/* Copies 4~7 bytes.  */
L(tail4):
	bf	29,L(tail2)
	lwz	6,0(src)
	stw	6,0(dst)
	bf	30,L(tail5)
	lhz	7,4(src)
	sth	7,4(dst)
	bflr	31
	lbz	8,6(src)
	stb	8,6(dst)
	/* Return original DST pointer.  */
	blr

	.align	4
/* Copies 2~3 bytes.  */
L(tail2):
	bf	30,1f
	lhz	6,0(src)
	sth	6,0(dst)
	bflr	31
	lbz	7,2(src)
	stb	7,2(dst)
	blr

	.align	4
L(tail5):
	bflr	31
	lbz	6,4(src)
	stb	6,4(dst)
	blr

	.align	4
1:
	bflr	31
	lbz	6,0(src)
	stb	6,0(dst)
	/* Return original DST pointer.  */
	blr


/* Handles copies of 0~8 bytes.  */
	.align	4
L(copy_LE_8):
	bne	cr6,L(tail4)

	/* Though we could've used ld/std here, they are still
	slow for unaligned cases.  */

	lwz	6,0(src)
	lwz	7,4(src)
	stw	6,0(dst)
	stw	7,4(dst)
	blr


/* Handle copies of 32+ bytes where DST is aligned (to quadword) but
   SRC is not.	Use aligned quadword loads from SRC, shifted to realign
   the data, allowing for aligned DST stores.  */
	.align	4
L(copy_GE_32_unaligned):
	clrldi	0,0,60	      /* Number of bytes until the 1st dst quadword.  */
	srdi	9,cnt,4	      /* Number of full quadwords remaining.  */

	beq	L(copy_GE_32_unaligned_cont)

	/* DST is not quadword aligned, get it aligned.  */

	mtocrf	0x01,0
	subf	cnt,0,cnt

	/* Vector instructions work best when proper alignment (16-bytes)
	is present.  Move 0~15 bytes as needed to get DST quadword-aligned.  */
1:
	bf	31,2f
	lbz	6,0(src)
	addi	src,src,1
	stb	6,0(dst)
	addi	dst,dst,1
2:
	bf	30,4f
	lhz	6,0(src)
	addi	src,src,2
	sth	6,0(dst)
	addi	dst,dst,2
4:
	bf	29,8f
	lwz	6,0(src)
	addi	src,src,4
	stw	6,0(dst)
	addi	dst,dst,4
8:
	bf	28,0f
	ld	6,0(src)
	addi	src,src,8
	std	6,0(dst)
	addi	dst,dst,8
0:
	srdi	9,cnt,4	      /* Number of full quadwords remaining.  */

	/* The proper alignment is present, it is OK to copy the bytes now.  */
L(copy_GE_32_unaligned_cont):

	/* Setup two indexes to speed up the indexed vector operations.  */
	clrldi	10,cnt,60
	li	6,16	      /* Index for 16-bytes offsets.  */
	li	7,32	      /* Index for 32-bytes offsets.  */
	cmpldi	cr1,10,0
	srdi	8,cnt,5	      /* Setup the loop counter.  */
	mtocrf	0x01,9
	cmpldi	cr6,9,1
#ifdef __LITTLE_ENDIAN__
	lvsr	5,0,src
#else
	lvsl	5,0,src
#endif
	lvx	3,0,src
	li	0,0
	bf	31,L(setup_unaligned_loop)

	/* Copy another 16 bytes to align to 32-bytes due to the loop.  */
	lvx	4,src,6
#ifdef __LITTLE_ENDIAN__
	vperm	6,4,3,5
#else
	vperm	6,3,4,5
#endif
	addi	src,src,16
	stvx	6,0,dst
	addi	dst,dst,16
	vor	3,4,4
	clrrdi	0,src,60

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

	lvx	4,src,6
#ifdef __LITTLE_ENDIAN__
	vperm	6,4,3,5
#else
	vperm	6,3,4,5
#endif
	lvx	3,src,7
#ifdef __LITTLE_ENDIAN__
	vperm	10,3,4,5
#else
	vperm	10,4,3,5
#endif
	addi	src,src,32
	stvx	6,0,dst
	stvx	10,dst,6
	addi	dst,dst,32
	bdnz	L(unaligned_loop)

	clrrdi	0,src,60

	.align	4
L(end_unaligned_loop):

	/* Check for tail bytes.  */
	mtocrf	0x01,cnt
	beqlr	cr1

	add	src,src,0

	/*  We have 1~15 tail bytes to copy, and DST is quadword aligned.  */
	/* Copy 8 bytes.  */
	bf	28,4f
	lwz	6,0(src)
	lwz	7,4(src)
	addi	src,src,8
	stw	6,0(dst)
	stw	7,4(dst)
	addi	dst,dst,8
4:	/* Copy 4~7 bytes.  */
	bf	29,L(tail2)
	lwz	6,0(src)
	stw	6,0(dst)
	bf	30,L(tail5)
	lhz	7,4(src)
	sth	7,4(dst)
	bflr	31
	lbz	8,6(src)
	stb	8,6(dst)
	/* Return original DST pointer.  */
	blr

END_GEN_TB (MEMCPY,TB_TOCLESS)
libc_hidden_builtin_def (memcpy)
