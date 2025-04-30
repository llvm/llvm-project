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

	.section .rodata.str1.4,"aMS",@progbits,1
	.align 2
.LC0:
	.string "longjmp causes uninitialized stack frame"
	.text

#define __longjmp_symbol ____longjmp_chk

#ifdef PIC
#  define LOAD_ARG \
	SETUP_GOT_ACCESS(r3,got_label2);			\
	addis	r3,r3,_GLOBAL_OFFSET_TABLE_-got_label2@ha;	\
	addi	r3,r3,_GLOBAL_OFFSET_TABLE_-got_label2@l;	\
	lwz	r3,.LC0@got(r3)
#else
# define LOAD_ARG \
	lis	r3,.LC0@ha;				\
	la	r3,.LC0@l(r3)
#endif

#define CHECK_SP(reg) \
	cmplw	reg, r1;				\
	bge+	.Lok;					\
	mflr	r0;					\
	stwu	r1,-32(r1);				\
	cfi_remember_state;				\
	cfi_adjust_cfa_offset (32);			\
	stw	r0,36(r1);				\
	cfi_offset (lr, 4);				\
	mr	r31,r3;					\
	mr	r30,r4;					\
	li	r3,0;					\
	addi	r4,r1,8;				\
	li	r0,__NR_sigaltstack;			\
	sc;						\
	/* Without working sigaltstack we cannot perform the test.  */ \
	bso	.Lok2;					\
	lwz	r0,12(r1);				\
	andi.	r3,r0,1;				\
	beq	.Lfail;					\
	lwz	r0,16(r1);				\
	lwz	r3,8(r1);				\
	add	r3,r3,r0;				\
	sub	r3,r3,reg;				\
	cmplw	r3,r0;					\
	bge+	.Lok2;					\
.Lfail:							\
	LOAD_ARG;					\
	bl	HIDDEN_JUMPTARGET (__fortify_fail);	\
.Lok2:							\
	mr	r3,r31;					\
	mr	r4,r30;					\
	cfi_restore_state;				\
.Lok:

#include <__longjmp-common.S>
