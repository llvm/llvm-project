/* Copyright (C) 2009-2021 Free Software Foundation, Inc.
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
#include <rtld-global-offsets.h>

	.section .rodata.str1.8,"aMS",@progbits,1
	.align 3
.LC0:
	.string "longjmp causes uninitialized stack frame"
	.section .toc,"aw"
.LC1:
	.tc .LC0[TC],.LC0
	.text

#define __longjmp ____longjmp_chk

#define CHECK_SP(reg) \
	cmpld	reg, r1;				\
	bge+	.Lok;					\
	mflr	r0;					\
	std	r0,FRAME_LR_SAVE(r1);			\
	mr	r31,r3;					\
	mr	r30,r4;					\
	stdu	r1,-FRAME_MIN_SIZE-32(r1);		\
	cfi_remember_state;				\
	cfi_adjust_cfa_offset (FRAME_MIN_SIZE+32);	\
	cfi_offset (lr, FRAME_LR_SAVE);			\
	li	r3,0;					\
	addi	r4,r1,FRAME_MIN_SIZE;			\
	li	r0,__NR_sigaltstack;			\
	sc;						\
	/* Without working sigaltstack we cannot perform the test.  */ \
	bso	.Lok2;					\
	lwz	r0,FRAME_MIN_SIZE+8(r1);		\
	andi.	r4,r0,1;				\
	beq	.Lfail;					\
	ld	r0,FRAME_MIN_SIZE+16(r1);		\
	ld	r4,FRAME_MIN_SIZE(r1);			\
	add	r4,r4,r0;				\
	sub	r3,r3,reg;				\
	cmpld	r3,r0;					\
	bge+	.Lok2;					\
.Lfail:							\
	ld	r3,.LC1@toc(2);				\
	bl	HIDDEN_JUMPTARGET (__fortify_fail);	\
	nop;						\
.Lok2:							\
	mr	r3,r31;					\
	mr	r4,r30;					\
	cfi_restore_state;				\
.Lok:

#include <__longjmp-common.S>
