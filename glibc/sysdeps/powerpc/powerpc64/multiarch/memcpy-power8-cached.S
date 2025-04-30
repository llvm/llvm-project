/* Optimized memcpy implementation for cached memory on PowerPC64/POWER8.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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


/* __ptr_t [r3] memcpy (__ptr_t dst [r3], __ptr_t src [r4], size_t len [r5]);
   Returns 'dst'.  */

	.machine power8
ENTRY_TOCLESS (__memcpy_power8_cached, 5)
	CALL_MCOUNT 3

	cmpldi	cr7,r5,15
	bgt	cr7,L(ge_16)
	andi.	r9,r5,0x1
	mr	r9,r3
	beq	cr0,1f
	lbz	r10,0(r4)
	addi	r9,r3,1
	addi	r4,r4,1
	stb	r10,0(r3)
1:
	andi.	r10,r5,0x2
	beq	cr0,2f
	lhz	r10,0(r4)
	addi	r9,r9,2
	addi	r4,r4,2
	sth	r10,-2(r9)
2:
	andi.	r10,r5,0x4
	beq	cr0,3f
	lwz	r10,0(r4)
	addi	r9,9,4
	addi	r4,4,4
	stw	r10,-4(r9)
3:
	andi.	r10,r5,0x8
	beqlr	cr0
	ld	r10,0(r4)
	std	r10,0(r9)
	blr

	.align 4
L(ge_16):
	cmpldi	cr7,r5,32
	ble	cr7,L(ge_16_le_32)
	cmpldi	cr7,r5,64
	ble	cr7,L(gt_32_le_64)

	/* Align dst to 16 bytes.  */
	andi.	r9,r3,0xf
	mr	r12,r3
	beq	cr0,L(dst_is_align_16)
	lxvd2x	v0,0,r4
	subfic	r12,r9,16
	subf	r5,r12,r5
	add	r4,r4,r12
	add	r12,r3,r12
	stxvd2x	v0,0,r3
L(dst_is_align_16):
	cmpldi	cr7,r5,127
	ble	cr7,L(tail_copy)
	mr	r9,r12
	srdi	r10,r5,7
	li	r11,16
	li	r6,32
	li	r7,48
	mtctr	r10
	clrrdi	r0,r5,7

	/* Main loop, copy 128 bytes each time.  */
	.align 4
L(copy_128):
	lxvd2x	v10,0,r4
	lxvd2x	v11,r4,r11
	addi	r8,r4,64
	addi	r10,r9,64
	lxvd2x	v12,r4,r6
	lxvd2x	v0,r4,r7
	addi	r4,r4,128
	stxvd2x v10,0,r9
	stxvd2x v11,r9,r11
	stxvd2x v12,r9,r6
	stxvd2x v0,r9,r7
	addi	r9,r9,128
	lxvd2x	v10,0,r8
	lxvd2x	v11,r8,r11
	lxvd2x	v12,r8,r6
	lxvd2x	v0,r8,r7
	stxvd2x v10,0,r10
	stxvd2x v11,r10,r11
	stxvd2x v12,r10,r6
	stxvd2x v0,r10,r7
	bdnz	L(copy_128)

	add	r12,r12,r0
	rldicl 	r5,r5,0,57
L(tail_copy):
	cmpldi	cr7,r5,63
	ble	cr7,L(tail_le_64)
	li	r8,16
	li	r10,32
	lxvd2x	v10,0,r4
	li	r9,48
	addi	r5,r5,-64
	lxvd2x	v11,r4,r8
	lxvd2x	v12,r4,r10
	lxvd2x	v0,r4,r9
	addi	r4,r4,64
	stxvd2x	v10,0,r12
	stxvd2x	v11,r12,r8
	stxvd2x	v12,r12,r10
	stxvd2x	v0,r12,9
	addi	r12,r12,64

L(tail_le_64):
	cmpldi	cr7,r5,32
	bgt	cr7,L(tail_gt_32_le_64)
	cmpdi	cr7,r5,0
	beqlr	cr7
	addi	r5,r5,-32
	li	r9,16
	add	r8,r4,r5
	add	r10,r12,r5
	lxvd2x	v12,r4,r5
	lxvd2x	v0,r8,r9
	stxvd2x	v12,r12,r5
	stxvd2x	v0,r10,r9
	blr

	.align 4
L(ge_16_le_32):
	addi	r5,r5,-16
	lxvd2x	v0,0,r4
	lxvd2x	v1,r4,r5
	stxvd2x	v0,0,r3
	stxvd2x	v1,r3,r5
	blr

	.align 4
L(gt_32_le_64):
	mr	r12,r3

	.align 4
L(tail_gt_32_le_64):
	li	r9,16
	lxvd2x	v0,0,r4
	addi	r5,r5,-32
	lxvd2x	v1,r4,r9
	add	r8,r4,r5
	lxvd2x	v2,r4,r5
	add	r10,r12,r5
	lxvd2x	v3,r8,r9
	stxvd2x	v0,0,r12
	stxvd2x	v1,r12,r9
	stxvd2x	v2,r12,r5
	stxvd2x	v3,r10,r9
	blr

END_GEN_TB (__memcpy_power8_cached,TB_TOCLESS)
