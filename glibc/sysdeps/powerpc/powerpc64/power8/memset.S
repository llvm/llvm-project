/* Optimized memset implementation for PowerPC64/POWER8.
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

/* void * [r3] memset (void *s [r3], int c [r4], size_t n [r5]));
   Returns 's'.  */

#ifndef MEMSET
# define MEMSET memset
#endif
	.machine  power8
ENTRY_TOCLESS (MEMSET, 5)
	CALL_MCOUNT 3

L(_memset):
	cmpldi	cr7,r5,31
	neg	r0,r3
	mr	r10,r3

	insrdi	r4,r4,8,48
	insrdi	r4,r4,16,32	/* Replicate byte to word.  */
	ble	cr7,L(write_LT_32)

	andi.	r11,r10,15	/* Check alignment of DST.  */
	insrdi	r4,r4,32,0	/* Replicate word to double word.  */

	beq	L(big_aligned)

	mtocrf	0x01,r0
	clrldi	r0,r0,60

	/* Get DST aligned to 16 bytes.  */
1:	bf	31,2f
	stb	r4,0(r10)
	addi	r10,r10,1

2:	bf	30,4f
	sth	r4,0(r10)
	addi	r10,r10,2

4:	bf	29,8f
	stw	r4,0(r10)
	addi	r10,r10,4

8:	bf      28,16f
	std     r4,0(r10)
	addi    r10,r10,8

16:	subf	r5,r0,r5

	.align	4
L(big_aligned):
	/* For sizes larger than 255 two possible paths:
	   - if constant is '0', zero full cache lines with dcbz
	   - otherwise uses vector instructions.  */
	cmpldi	cr5,r5,255
	dcbtst	0,r10
	cmpldi	cr6,r4,0
	crand	27,26,21
	bt	27,L(huge_dcbz)
	bge	cr5,L(huge_vector)


	/* Size between 32 and 255 bytes with constant different than 0, use
	   doubleword store instruction to achieve best throughput.  */
	srdi    r8,r5,5
	clrldi  r11,r5,59
	cmpldi  cr6,r11,0
	cmpdi	r8,0
	beq     L(tail_bytes)
	mtctr   r8

	/* Main aligned write loop, writes 32-bytes at a time.  */
	.align  4
L(big_loop):
	std     r4,0(r10)
	std     r4,8(r10)
	std     r4,16(r10)
	std     r4,24(r10)
	addi    r10,r10,32
	bdz     L(tail_bytes)

	std     r4,0(r10)
	std     r4,8(r10)
	std     r4,16(r10)
	std     r4,24(r10)
	addi    r10,10,32
	bdnz    L(big_loop)

	b       L(tail_bytes)

	/* Write remaining 1~31 bytes.  */
	.align  4
L(tail_bytes):
	beqlr   cr6

	srdi    r7,r11,4
	clrldi  r8,r11,60
	mtocrf  0x01,r7

	.align	4
	bf	31,8f
	std	r4,0(r10)
	std	r4,8(r10)
	addi	r10,r10,16

	.align	4
8:	mtocrf	0x1,r8
	bf	28,4f
	std	r4,0(r10)
	addi	r10,r10,8

	.align	4
4:	bf      29,2f
	stw     4,0(10)
	addi    10,10,4

	.align 	4
2:	bf      30,1f
	sth     4,0(10)
	addi    10,10,2

	.align  4
1:      bflr    31
	stb     4,0(10)
	blr

	/* Size larger than 255 bytes with constant different than 0, use
	   vector instruction to achieve best throughput.  */
L(huge_vector):
	/* Replicate set byte to quadword in VMX register.  */
	mtvsrd	v1,r4
	xxpermdi 32,v0,v1,0
	vspltb	 v2,v0,15

	/* Main aligned write loop: 128 bytes at a time.  */
	li	r6,16
	li	r7,32
	li	r8,48
	mtocrf	0x02,r5
	srdi	r12,r5,7
	cmpdi	r12,0
	beq	L(aligned_tail)
	mtctr	r12
	b	L(aligned_128loop)

	.align  4
L(aligned_128loop):
	stvx	v2,0,r10
	stvx	v2,r10,r6
	stvx	v2,r10,r7
	stvx	v2,r10,r8
	addi	r10,r10,64
	stvx	v2,0,r10
	stvx	v2,r10,r6
	stvx	v2,r10,r7
	stvx	v2,r10,r8
	addi	r10,r10,64
	bdnz	L(aligned_128loop)

	/* Write remaining 1~127 bytes.  */
L(aligned_tail):
	mtocrf	0x01,r5
	bf	25,32f
	stvx	v2,0,r10
	stvx	v2,r10,r6
	stvx	v2,r10,r7
	stvx	v2,r10,r8
	addi	r10,r10,64

32:	bf	26,16f
	stvx	v2,0,r10
	stvx	v2,r10,r6
	addi	r10,r10,32

16:	bf	27,8f
	stvx	v2,0,r10
	addi	r10,r10,16

8:	bf	28,4f
	std     r4,0(r10)
	addi	r10,r10,8

	/* Copies 4~7 bytes.  */
4:	bf	29,L(tail2)
	stw     r4,0(r10)
	bf      30,L(tail5)
	sth     r4,4(r10)
	bflr	31
	stb     r4,6(r10)
	/* Return original DST pointer.  */
	blr

	/* Special case when value is 0 and we have a long length to deal
	   with.  Use dcbz to zero out a full cacheline of 128 bytes at a time.
	   Before using dcbz though, we need to get the destination 128-byte
	   aligned.  */
	.align	4
L(huge_dcbz):
	andi.	r11,r10,127
	neg	r0,r10
	beq	L(huge_dcbz_aligned)

	clrldi	r0,r0,57
	subf	r5,r0,r5
	srdi	r0,r0,3
	mtocrf	0x01,r0

	/* Write 1~128 bytes until DST is aligned to 128 bytes.  */
8:	bf	28,4f

	std	r4,0(r10)
	std	r4,8(r10)
	std	r4,16(r10)
	std	r4,24(r10)
	std	r4,32(r10)
	std	r4,40(r10)
	std	r4,48(r10)
	std	r4,56(r10)
	addi	r10,r10,64

	.align	4
4:	bf	29,2f
	std	r4,0(r10)
	std	r4,8(r10)
	std	r4,16(r10)
	std	r4,24(r10)
	addi	r10,r10,32

	.align	4
2:	bf	30,1f
	std	r4,0(r10)
	std	r4,8(r10)
	addi	r10,r10,16

	.align	4
1:	bf	31,L(huge_dcbz_aligned)
	std	r4,0(r10)
	addi	r10,r10,8

L(huge_dcbz_aligned):
	/* Setup dcbz unroll offsets and count numbers.  */
	srdi	r8,r5,9
	clrldi	r11,r5,55
	cmpldi	cr6,r11,0
	li	r9,128
	cmpdi	r8,0
	beq     L(huge_tail)
	li	r7,256
	li	r6,384
	mtctr	r8

	.align	4
L(huge_loop):
	/* Sets 512 bytes to zero in each iteration, the loop unrolling shows
	   a throughput boost for large sizes (2048 bytes or higher).  */
	dcbz	0,r10
	dcbz	r9,r10
	dcbz	r7,r10
	dcbz	r6,r10
	addi	r10,r10,512
	bdnz	L(huge_loop)

	beqlr	cr6

L(huge_tail):
	srdi    r6,r11,8
	srdi    r7,r11,4
	clrldi  r8,r11,4
	cmpldi  cr6,r8,0
	mtocrf  0x01,r6

	beq	cr6,L(tail)

	/* We have 1~511 bytes remaining.  */
	.align	4
32:	bf	31,16f
	dcbz	0,r10
	dcbz	r9,r10
	addi	r10,r10,256

	.align	4
16:	mtocrf  0x01,r7
	bf	28,8f
	dcbz	0,r10
	addi	r10,r10,128

	.align 	4
8:	bf	29,4f
	std	r4,0(r10)
	std	r4,8(r10)
	std	r4,16(r10)
	std	r4,24(r10)
	std	r4,32(r10)
	std	r4,40(r10)
	std	r4,48(r10)
	std	r4,56(r10)
	addi	r10,r10,64

	.align	4
4:	bf	30,2f
	std	r4,0(r10)
	std	r4,8(r10)
	std	r4,16(r10)
	std	r4,24(r10)
	addi	r10,r10,32

	.align	4
2:	bf	31,L(tail)
	std	r4,0(r10)
	std	r4,8(r10)
	addi	r10,r10,16
	.align	4

	/* Remaining 1~15 bytes.  */
L(tail):
	mtocrf  0x01,r8

	.align
8:	bf	28,4f
	std	r4,0(r10)
	addi	r10,r10,8

	.align	4
4:	bf	29,2f
	stw	r4,0(r10)
	addi	r10,r10,4

	.align	4
2:	bf	30,1f
	sth	r4,0(r10)
	addi	r10,r10,2

	.align	4
1:	bflr	31
	stb	r4,0(r10)
	blr

	/* Handle short copies of 0~31 bytes.  Best throughput is achieved
	   by just unrolling all operations.  */
	.align	4
L(write_LT_32):
	cmpldi	cr6,5,8
	mtocrf	0x01,r5
	ble	cr6,L(write_LE_8)

	/* At least 9 bytes to go.  */
	neg	r8,r4
	andi.	r0,r8,3
	cmpldi	cr1,r5,16
	beq	L(write_LT_32_aligned)

	/* Force 4-byte alignment for SRC.  */
	mtocrf	0x01,r0
	subf	r5,r0,r5

2:	bf	30,1f
	/* Use stb instead of sth because it doesn't generate
	   alignment interrupts on cache-inhibited storage.  */
	stb	r4,0(r10)
	stb	r4,1(r10)
	addi	r10,r10,2

1:	bf	31,L(end_4bytes_alignment)
	stb	r4,0(r10)
	addi	r10,r10,1

	.align	4
L(end_4bytes_alignment):
	cmpldi	cr1,r5,16
	mtocrf	0x01,r5

L(write_LT_32_aligned):
	blt	cr1,8f

	stw	r4,0(r10)
	stw	r4,4(r10)
	stw	r4,8(r10)
	stw	r4,12(r10)
	addi	r10,r10,16

8:	bf	28,L(tail4)
	stw	r4,0(r10)
	stw	r4,4(r10)
	addi	r10,r10,8

	.align	4
	/* Copies 4~7 bytes.  */
L(tail4):
	bf	29,L(tail2)
	stw	r4,0(r10)
	bf	30,L(tail5)
	sth	r4,4(r10)
	bflr	31
	stb	r4,6(r10)
	blr

	.align	4
	/* Copies 2~3 bytes.  */
L(tail2):
	bf	30,1f
	sth	r4,0(r10)
	bflr	31
	stb	r4,2(r10)
	blr

	.align	4
L(tail5):
	bflr	31
	stb	r4,4(r10)
	blr

	.align	4
1: 	bflr	31
	stb	r4,0(r10)
	blr

	/* Handles copies of 0~8 bytes.  */
	.align	4
L(write_LE_8):
	bne	cr6,L(LE7_tail4)
	/* If input is word aligned, use stw, else use stb.  */
	andi.	r0,r10,3
	bne	L(8_unalign)

	stw	r4,0(r10)
	stw	r4,4(r10)
	blr

	/* Unaligned input and size is 8.  */
	.align	4
L(8_unalign):
	andi.	r0,r10,1
	beq	L(8_hwalign)
	stb	r4,0(r10)
	sth	r4,1(r10)
	sth	r4,3(r10)
	sth	r4,5(r10)
	stb	r4,7(r10)
	blr

	/* Halfword aligned input and size is 8.  */
	.align	4
L(8_hwalign):
	sth	r4,0(r10)
	sth	r4,2(r10)
	sth	r4,4(r10)
	sth	r4,6(r10)
	blr

	.align	4
	/* Copies 4~7 bytes.  */
L(LE7_tail4):
	/* Use stb instead of sth because it doesn't generate
	   alignment interrupts on cache-inhibited storage.  */
	bf	29,L(LE7_tail2)
	stb	r4,0(r10)
	stb	r4,1(r10)
	stb	r4,2(r10)
	stb	r4,3(r10)
	bf	30,L(LE7_tail5)
	stb	r4,4(r10)
	stb	r4,5(r10)
	bflr	31
	stb	r4,6(r10)
	blr

	.align	4
	/* Copies 2~3 bytes.  */
L(LE7_tail2):
	bf	30,1f
	stb	r4,0(r10)
	stb	r4,1(r10)
	bflr	31
	stb	r4,2(r10)
	blr

	.align	4
L(LE7_tail5):
	bflr	31
	stb	r4,4(r10)
	blr

	.align	4
1: 	bflr	31
	stb	r4,0(r10)
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
