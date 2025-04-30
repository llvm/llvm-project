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
	/* Unreachable.	 */				\
.Lfail0:						\
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
	cmp/hs	r15, reg;				\
	bt	.Lok;					\
	CALL_FAIL					\
.Lok:

#include <__longjmp.S>
