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

#include <sigaltstack-offsets.h>

	.section .rodata.str1.1,"aMS",@progbits,1
	.type	longjmp_msg,@object
longjmp_msg:
	.string "longjmp causes uninitialized stack frame"
	.size	longjmp_msg, .-longjmp_msg
	.text

#define __longjmp ____longjmp_chk

#ifdef PIC
# define CALL_FAIL \
	mov.l	.Lfail, r1;				\
	mov.l	.Lstr, r4;				\
	mov.l	r12, @-r15;				\
	cfi_remember_state;				\
	cfi_adjust_cfa_offset (4);			\
	cfi_rel_offset (r12, 0);			\
	mova	.Lgot, r0;				\
	mov.l	.Lgot, r12;				\
	add	r0, r12;				\
	sts.l	pr, @-r15;				\
	cfi_adjust_cfa_offset (4);			\
	cfi_rel_offset (pr, 0);				\
	bsrf	r1;					\
	 add	r12, r4;				\
.Lfail0:						\
	/* Unreachable.	 */				\
	.align	2;					\
.Lgot:							\
	.long	_GLOBAL_OFFSET_TABLE_;			\
.Lstr:							\
	.long	longjmp_msg@GOTOFF;			\
.Lfail:							\
	.long	__GI___fortify_fail@PLT-(.Lfail0-.);	\
	cfi_restore_state;
#else
# define CALL_FAIL \
	mov.l	.Lfail, r1;				\
	mov.l	.Lstr, r4;				\
	sts.l	pr, @-r15;				\
	cfi_remember_state;				\
	cfi_adjust_cfa_offset (4);			\
	cfi_rel_offset (pr, 0);				\
	jsr	@r1;					\
	 nop;						\
	/* Unreachable.	 */				\
	.align	2;					\
.Lstr:							\
	.long	longjmp_msg;				\
.Lfail:							\
	.long	__fortify_fail;				\
	cfi_restore_state;
#endif

#define CHECK_SP(reg) \
	/* Jumping to a higher-address frame is always allowed.	 */	\
	cmp/hs	r15, reg;						\
	bt	.Lok;							\
									\
	mov.l	r0, @-r15; /* The return value is already in here.  */	\
	cfi_adjust_cfa_offset (4);					\
	mov.l	r1, @-r15; /* PTR_DEMANGLE helper.  */			\
	cfi_adjust_cfa_offset (4);					\
	mov.l	r2, @-r15; /* The new SP value is already in here.  */	\
	cfi_adjust_cfa_offset (4);					\
	mov.l	r4, @-r15; /* We'll still need this one.  */		\
	cfi_adjust_cfa_offset (4);					\
	add	#-sizeSS, r15;						\
	cfi_adjust_cfa_offset (sizeSS);					\
	mov	#0, r4;							\
	mov	r15, r5;						\
	DO_CALL (sigaltstack, 2);					\
	/* Without working sigaltstack we cannot perform the test.  */	\
	tst	r0, r0;							\
	bf	.Lok2;							\
	mov.l	@(oSS_FLAGS, r15), r0;					\
	tst	#SS_ONSTACK, r0;					\
	bt	.Lcall_fail;						\
	mov.l	@(oSS_SIZE, r15), r2;					\
	mov.l	@(oSS_SP, r15), r1;					\
	add	r2, r1;							\
	sub	r8, r1;							\
	cmp/hi	r1, r2;							\
	bf	.Lok2;							\
.Lcall_fail:								\
	CALL_FAIL							\
									\
.Lok2:									\
	add	#sizeSS, r15;						\
	cfi_adjust_cfa_offset (-sizeSS);				\
	mov.l	@r15+, r4;						\
	cfi_adjust_cfa_offset (-4);					\
	mov.l	@r15+, r2;						\
	cfi_adjust_cfa_offset (-4);					\
	mov.l	@r15+, r1;						\
	cfi_adjust_cfa_offset (-4);					\
	mov.l	@r15+, r0;						\
	cfi_adjust_cfa_offset (-4);					\
.Lok:

#include <__longjmp.S>
