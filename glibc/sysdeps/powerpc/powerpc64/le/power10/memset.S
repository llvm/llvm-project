/* Optimized memset implementation for POWER10 LE.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

	.machine  power9
ENTRY_TOCLESS (MEMSET, 5)
	CALL_MCOUNT 3

L(_memset):
	/* Assume memset of zero length is uncommon, and just let it go
	   through the small path below.  */
	cmpldi	r5,64

	/* Replicate byte to quad word.  */
	mtvsrd	v0+32,r4
	vspltb	v0,v0,7

	li	r7,16
	sldi	r8,r7,56

	bgt	L(large)

	/* For short lengths we want to avoid as many branches as possible.
	   We use store VSX vector with length instructions to do this.
	   It takes advantage of the fact that if the length passed to stxvl
	   is zero nothing is done, effectively a no-op.  */
	sldi	r5,r5,56

	addi	r10,r3,16

	sub.	r11,r5,r8
	isellt	r11,0,r11	/* Saturate the subtraction to zero.  */

	stxvl	v0+32,r3,r5
	stxvl	v0+32,r10,r11

	addi	r9,r3,32
	addi	r10,r3,48

	sub.	r11,r11,r8
	isellt	r11,0,r11

	sub.	r5,r11,r8
	isellt	r5,0,r5

	stxvl	v0+32,r9,r11
	stxvl	v0+32,r10,r5

	blr

	.balign	16
L(large):
	mr	r6,r3	/* Don't modify r3 since we need to return it.  */

	/* Get dest 16B aligned.  */
	neg	r0,r3
	clrldi.	r7,r0,(64-4)
	beq	L(aligned)
	rldic	r9,r0,56,4	/* (~X & 0xf)<<56 "clrlsldi r9,r0,64-4,56".  */

	stxvl	v0+32,r6,r9	/* Store up to 15B until aligned address.  */

	add	r6,r6,r7
	sub	r5,r5,r7

	/* Go to tail if there is less than 64B left after alignment.  */
	cmpldi	r5,64
	blt	L(tail_64)

	.balign	16
L(aligned):
	/* Go to tail if there is less than 128B left after alignment.  */
	srdi.	r0,r5,7
	beq	L(tail_128)

	/* If c == 0 && n >= 256 use dcbz to zero out full cache blocks.  */
	cmpldi	cr5,r5,255
	cmpldi	cr6,r4,0
	crand	27,26,21
	bt	27,L(dcbz)

	mtctr	r0

	.balign	32
L(loop):
	stxv	v0+32,0(r6)
	stxv	v0+32,16(r6)
	stxv	v0+32,32(r6)
	stxv	v0+32,48(r6)
	stxv	v0+32,64(r6)
	stxv	v0+32,80(r6)
	stxv	v0+32,96(r6)
	stxv	v0+32,112(r6)
	addi	r6,r6,128
	bdnz	L(loop)

	.balign	16
L(tail):
	/* 127B or less left, finish the tail or return.  */
	andi.	r5,r5,127
	beqlr

	cmpldi	r5,64
	blt	L(tail_64)

	.balign	16
L(tail_128):
	/* Stores a minimum of 64B and up to 128B and return.  */
	stxv	v0+32,0(r6)
	stxv	v0+32,16(r6)
	stxv	v0+32,32(r6)
	stxv	v0+32,48(r6)
	addi	r6,r6,64
	andi.	r5,r5,63
	beqlr

	.balign	16
L(tail_64):
	/* Stores up to 64B and return.  */
	sldi	r5,r5,56

	addi	r10,r6,16

	sub.	r11,r5,r8
	isellt	r11,0,r11

	stxvl	v0+32,r6,r5
	stxvl	v0+32,r10,r11

	sub.	r11,r11,r8
	blelr

	addi	r9,r6,32
	addi	r10,r6,48

	isellt	r11,0,r11

	sub.	r5,r11,r8
	isellt	r5,0,r5

	stxvl	v0+32,r9,r11
	stxvl	v0+32,r10,r5

	blr

	.balign	16
L(dcbz):
	/* Special case when value is 0 and we have a long length to deal
	   with.  Use dcbz to zero out a full cacheline of 128 bytes at a time.
	   Before using dcbz though, we need to get the destination 128-byte
	   aligned.  */
	neg	r0,r6
	clrldi.	r0,r0,(64-7)
	beq	L(dcbz_aligned)

	sub	r5,r5,r0
	mtocrf	0x2,r0	/* copying bits 57..59 to cr6. The ones for sizes 64,
			   32 and 16 which need to be checked.  */

	/* Write 16-128 bytes until DST is aligned to 128 bytes.  */
64:	bf	25,32f
	stxv	v0+32,0(r6)
	stxv	v0+32,16(r6)
	stxv	v0+32,32(r6)
	stxv	v0+32,48(r6)
	addi	r6,r6,64

32:	bf	26,16f
	stxv	v0+32,0(r6)
	stxv	v0+32,16(r6)
	addi	r6,r6,32

16:	bf	27,L(dcbz_aligned)
	stxv	v0+32,0(r6)
	addi	r6,r6,16

	.balign	16
L(dcbz_aligned):
	/* Setup dcbz unroll offsets and count numbers.  */
	srdi.	r0,r5,9
	li	r9,128
	beq	L(bcdz_tail)
	li	r10,256
	li	r11,384
	mtctr	r0

	.balign	16
L(dcbz_loop):
	/* Sets 512 bytes to zero in each iteration, the loop unrolling shows
	   a throughput boost for large sizes (2048 bytes or higher).  */
	dcbz	0,r6
	dcbz	r9,r6
	dcbz	r10,r6
	dcbz	r11,r6
	addi	r6,r6,512
	bdnz	L(dcbz_loop)

	andi.	r5,r5,511
	beqlr

	.balign	16
L(bcdz_tail):
	/* We have 1-511 bytes remaining.  */
	srdi.	r0,r5,7
	beq	L(tail)

	mtocrf	0x1,r0

256:	bf	30,128f
	dcbz	0,r6
	dcbz	r9,r6
	addi	r6,r6,256

128:	bf	31,L(tail)
	dcbz	0,r6
	addi	r6,r6,128

	b	L(tail)

END_GEN_TB (MEMSET,TB_TOCLESS)
libc_hidden_builtin_def (memset)

/* Copied from bzero.S to prevent the linker from inserting a stub
   between bzero and memset.  */
ENTRY_TOCLESS (__bzero)
	CALL_MCOUNT 2
	mr	r5,r4
	li	r4,0
	b	L(_memset)
END (__bzero)
#ifndef __bzero
weak_alias (__bzero, bzero)
#endif
