/* longjmp with sp check for C-SKY ABIV2.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

	.section .rodata.str1.8,"aMS",@progbits,1
	.align  2
	.type   longjmp_msg,@object
longjmp_msg:
	.string "longjmp causes uninitialized stack frame"
	.size   longjmp_msg, .-longjmp_msg

	.text
#define __longjmp ____longjmp_chk

#ifdef __PIC__
# define CALL_FAIL	\
	subi	sp, 8;						\
	stw	gb, (sp, 0);					\
	grs	gb, .Lgetpc1;					\
.Lgetpc1:							\
	lrw	t0, .Lgetpc1@GOTPC;				\
	addu	gb, gb, t0;					\
	lrw	a0, longjmp_msg@GOTOFF;				\
	addu	a0, a0, gb;					\
	lrw	t0, (HIDDEN_JUMPTARGET (__fortify_fail))@PLT;	\
	ldr.w	t0, (gb, t0 << 0);				\
	jsr	t0;						\
	ldw	gb, (sp, 0);					\
	addi	sp, 8;
#else /* __PIC__ */
# define CALL_FAIL	\
	lrw	a0, longjmp_msg;				\
	jsri	HIDDEN_JUMPTARGET (__fortify_fail);
#endif /* __PIC__ */

# define CHECK_SP(reg)	\
	cmplt	sp, reg;					\
	bt	.Lok1;						\
	mov	r9, a0;						\
	mov	t0, r7;						\
	lrw	r7, __NR_sigaltstack;				\
	movi	a0, 0;						\
	subi	sp, 12;	/* sizeof (stack_t) */			\
	mov	a1, sp;						\
	trap	0;						\
	mov	r7, t0;						\
	cmpnei	a0, 0;						\
	bt	.Lok;						\
	ldw	a1, (sp, 4);					\
	btsti	a1, 1;						\
	bt	.Lfail;						\
	ldw	a1, (sp, 0);					\
	ldw	a3, (sp, 8);					\
	add	a1, a3;						\
	sub	a1, reg;					\
	cmphs	a1, a3;						\
	bt	.Lok;						\
.Lfail:								\
	addi	sp, 12;						\
	CALL_FAIL						\
.Lok:								\
	mov	a0, r9;						\
.Lok1:

#include <__longjmp.S>
