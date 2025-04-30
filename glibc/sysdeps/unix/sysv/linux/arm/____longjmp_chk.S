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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

	.section .rodata.str1.1,"aMS",%progbits,1
	.type	longjmp_msg,%object
longjmp_msg:
	.string "longjmp causes uninitialized stack frame"
	.size	longjmp_msg, .-longjmp_msg
	.text

#define __longjmp ____longjmp_chk

#ifdef PIC
# define CALL_FAIL						\
	ldr	sl, .L_GOT;					\
	cfi_undefined (sl);					\
.L_GOT_OFF:							\
	add	sl, pc, sl;					\
	ldr	r0, .Lstr;					\
	add	r0, sl, r0;					\
	B	PLTJMP(HIDDEN_JUMPTARGET(__fortify_fail));	\
.L_GOT:								\
	.word	_GLOBAL_OFFSET_TABLE_-(.L_GOT_OFF+8);		\
.Lstr:								\
	.word	longjmp_msg(GOTOFF);
#else
# define CALL_FAIL					\
	ldr	r0, .Lstr;				\
	B	HIDDEN_JUMPTARGET(__fortify_fail);	\
.Lstr:							\
	.word	longjmp_msg;
#endif

#define CHECK_SP(reg)				\
	cfi_remember_state;			\
	cmp	sp, reg;			\
	bls	.Lok;				\
	push	{ r7 };				\
	cfi_adjust_cfa_offset (4);		\
	cfi_rel_offset (r7, 0);			\
	mov	r5, r0;				\
	cfi_undefined (r5);			\
	mov	r7, #SYS_ify(sigaltstack);	\
	mov	r0, #0;				\
	sub	sp, sp, #12; /* == sizeof (stack_t) */ \
	cfi_adjust_cfa_offset (12);		\
	cfi_remember_state;			\
	mov	r1, sp;				\
	swi	#0;				\
	cmp	r0, #0;				\
	bne	.Lok2;				\
	ldr	r1, [sp, #4];			\
	tst	r1, #1;				\
	beq	.Lfail;				\
	ldr	r2, [sp, #0];			\
	ldr	r3, [sp, #8];			\
	add	r2, r2, r3;			\
	sub	r2, r2, reg;			\
	cmp	r3, r2;				\
	bls	.Lok2;				\
.Lfail:						\
	add	sp, sp, #12;			\
	cfi_adjust_cfa_offset (-12);		\
	pop	{ r7 };				\
	cfi_adjust_cfa_offset (-4);		\
	cfi_restore (r7);			\
	CALL_FAIL				\
	cfi_restore_state;			\
.Lok2:						\
	mov	r0, r5;				\
	cfi_restore_state;			\
.Lok:

#include <__longjmp.S>
