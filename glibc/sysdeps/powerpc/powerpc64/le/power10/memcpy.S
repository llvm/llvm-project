/* Optimized memcpy implementation for POWER10.
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
   <http://www.gnu.org/licenses/>.  */

#include <sysdep.h>


#ifndef MEMCPY
# define MEMCPY memcpy
#endif

/* __ptr_t [r3] memcpy (__ptr_t dst [r3], __ptr_t src [r4], size_t len [r5]);
	   Returns 'dst'.  */

	.machine power9
ENTRY_TOCLESS (MEMCPY, 5)
	CALL_MCOUNT 3

	/* Copy up to 16 bytes.  */
	sldi	r6,r5,56	/* Prepare [l|st]xvl counter.  */
	lxvl	v10,r4,r6
	stxvl	v10,r3,r6
	subic.	r6,r5,16	/* Return if len <= 16.  */
	blelr

	/* If len >= 256, assume nothing got copied before and copy
	   again.  This might cause issues with overlapped memory, but memcpy
	   is not expected to treat overlapped memory.  */
	cmpdi	r5,256
	bge	L(copy_ge_256)
	/* 16 < len < 256 and the first 16 bytes have already been copied.  */
	addi	r10,r3,16	/* Keep r3 intact as return value.  */
	addi	r4,r4,16
	subi	r5,r5,16
	b	L(copy_lt_256)	/* Avoid the main loop if len < 256.  */

	.p2align 5
L(copy_ge_256):
	mr	r10,r3		/* Keep r3 intact as return value.  */
	/* Align dst to 16 bytes.  */
	andi.	r9,r10,0xf
	beq	L(dst_is_align_16)
	lxv	v10,0(r4)
	subfic	r12,r9,16
	subf	r5,r12,r5
	add	r4,r4,r12
	stxv	v10,0(r3)
	add	r10,r3,r12

L(dst_is_align_16):
	srdi	r9,r5,7		/* Divide by 128.  */
	mtctr	r9
	addi	r6,r4,64
	addi	r7,r10,64


	/* Main loop, copy 128 bytes per iteration.
	   Use r6=src+64 and r7=dest+64 in order to reduce the dependency on
	   r4 and r10.  */
	.p2align 5
L(copy_128):

	lxv	v10, 0(r4)
	lxv	v11, 16(r4)
	lxv	v12, 32(r4)
	lxv	v13, 48(r4)

	addi	r4,r4,128

	stxv	v10, 0(r10)
	stxv	v11, 16(r10)
	stxv	v12, 32(r10)
	stxv	v13, 48(r10)

	addi	r10,r10,128

	lxv	v10, 0(r6)
	lxv	v11, 16(r6)
	lxv	v12, 32(r6)
	lxv	v13, 48(r6)

	addi	r6,r6,128

	stxv	v10, 0(r7)
	stxv	v11, 16(r7)
	stxv	v12, 32(r7)
	stxv	v13, 48(r7)

	addi	r7,r7,128

	bdnz	L(copy_128)

	clrldi.	r5,r5,64-7	/* Have we copied everything?  */
	beqlr

	.p2align 5
L(copy_lt_256):
	cmpdi	r5,16
	ble	L(copy_le_16)
	srdi.	r9,r5,5		/* Divide by 32.  */
	beq	L(copy_lt_32)
	mtctr	r9
	/* Use r6=src+32, r7=dest+32, r8=src+64, r9=dest+64 in order to reduce
	   the dependency on r4 and r10.  */
	addi	r6,r4,32
	addi	r7,r10,32
	addi	r8,r4,64
	addi	r9,r10,64

	.p2align 5
	/* Copy 32 bytes at a time, unaligned.
	   The loop is unrolled 3 times in order to reduce the dependency on
	   r4 and r10, copying up-to 96 bytes per iteration.  */
L(copy_32):
	lxv	v10, 0(r4)
	lxv	v11, 16(r4)
	stxv	v10, 0(r10)
	stxv	v11, 16(r10)
	bdz	L(end_copy_32a)
	addi	r4,r4,96
	addi	r10,r10,96

	lxv	v10, 0(r6)
	lxv	v11, 16(r6)
	addi	r6,r6,96
	stxv	v10, 0(r7)
	stxv	v11, 16(r7)
	bdz	L(end_copy_32b)
	addi	r7,r7,96

	lxv	v12, 0(r8)
	lxv	v13, 16(r8)
	addi	r8,r8,96
	stxv	v12, 0(r9)
	stxv	v13, 16(r9)
	addi	r9,r9,96
	bdnz	L(copy_32)

	clrldi.	r5,r5,64-5	/* Have we copied everything?  */
	beqlr
	cmpdi	r5,16
	ble	L(copy_le_16)
	b	L(copy_lt_32)

	.p2align 5
L(end_copy_32a):
	clrldi.	r5,r5,64-5	/* Have we copied everything?  */
	beqlr
	/* 32 bytes have been copied since the last update of r4 and r10.  */
	addi	r4,r4,32
	addi	r10,r10,32
	cmpdi	r5,16
	ble	L(copy_le_16)
	b	L(copy_lt_32)

	.p2align 5
L(end_copy_32b):
	clrldi.	r5,r5,64-5	/* Have we copied everything?  */
	beqlr
	/* The last iteration of the loop copied 64 bytes.  Update r4 and r10
	   accordingly.  */
	addi	r4,r4,-32
	addi	r10,r10,-32
	cmpdi	r5,16
	ble	L(copy_le_16)

	.p2align 5
L(copy_lt_32):
	lxv	v10, 0(r4)
	stxv	v10, 0(r10)
	addi	r4,r4,16
	addi	r10,r10,16
	subi	r5,r5,16

	.p2align 5
L(copy_le_16):
	sldi	r6,r5,56
	lxvl	v10,r4,r6
	stxvl	v10,r10,r6
	blr


END_GEN_TB (MEMCPY,TB_TOCLESS)
libc_hidden_builtin_def (memcpy)
