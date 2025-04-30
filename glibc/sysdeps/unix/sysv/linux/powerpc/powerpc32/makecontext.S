/* Set up a context to call a function.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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
#include <shlib-compat.h>

#define __ASSEMBLY__
#include <asm/ptrace.h>
#include "ucontext_i.h"

ENTRY(__makecontext)
	/* Set up the first 7 args to the function in its registers */
	lwz	r11,_UC_REGS_PTR(r3)
	stw	r6,_UC_GREGS+(PT_R3*4)(r11)
	stw	r7,_UC_GREGS+(PT_R4*4)(r11)
	stw	r8,_UC_GREGS+(PT_R5*4)(r11)
	stw	r9,_UC_GREGS+(PT_R6*4)(r11)
	stw	r10,_UC_GREGS+(PT_R7*4)(r11)
	lwz	r8,8(r1)
	lwz	r9,12(r1)
	stw	r8,_UC_GREGS+(PT_R8*4)(r11)
	stw	r9,_UC_GREGS+(PT_R9*4)(r11)

	/* Set the NIP to the start of the function */
	stw	r4,_UC_GREGS+(PT_NIP*4)(r11)

	/* Set the function's r31 to ucp->uc_link for the exitcode below. */
	lwz	r7,_UC_LINK(r3)
	stw	r7,_UC_GREGS+(PT_R31*4)(r11)

	/* Set the function's LR to point to the exitcode below. */
#ifdef PIC
	mflr	r0
	cfi_register(lr,r0)
	/* Use this conditional form of branch and link to avoid destroying
	   the cpu link stack used to predict blr return addresses.  */
	bcl	20,31,1f
1:	mflr	r6
	addi	r6,r6,L(exitcode)-1b
	mtlr	r0
	cfi_same_value (lr)
#else
	lis	r6,L(exitcode)@ha
	addi	r6,r6,L(exitcode)@l
#endif
	stw	r6,_UC_GREGS+(PT_LNK*4)(r11)

	/*
	 * Set up the stack frame for the function.
	 * If we have more than 5 args to the function (8 args to makecontext),
	 * there will be some arguments on the stack which have to end up
	 * in registers.  If there are more than 8 args to the function,
	 * we have to copy (argc - 8) args from our stack to the functions'
	 * stack (and allow space for them in the frame).
	 */
	lwz	r4,_UC_STACK_SP(r3)
	lwz	r8,_UC_STACK_SIZE(r3)
	add	r4,r4,r8
	rlwinm	r4,r4,0,0,27	/* round down to 16-byte boundary */
	addi	r7,r4,-16	/* stack frame for fn's caller */
	cmpwi	r5,8
	blt	2f		/* less than 8 args is easy */
	lwz	r10,16(r1)
	stw	r10,_UC_GREGS+(PT_R10*4)(r11)
	beq	2f		/* if exactly 8 args */
	subi	r9,r5,3
	subi	r5,r5,8
	rlwinm	r9,r9,2,0,27
	subf	r7,r9,r4
	mtctr	r5		/* copy the 9th and following args */
	addi	r6,r1,16
	addi	r8,r7,4
3:	lwzu	r10,4(r6)
	stwu	r10,4(r8)
	bdnz	3b
2:	stw	r7,_UC_GREGS+(PT_R1*4)(r11)
	li	r6,0
	stw	r6,0(r7)

	blr

	cfi_endproc
	nop
/*
 * If the function returns, it comes here.  We put ucp->uc_link in
 * r31, which is a callee-saved register.  We have to continue with
 * the context that r31 points to, or exit if it is 0.
 */
L(exitcode):
	mr.	r3,r31
	beq	4f
	bl	__setcontext@local
4:	bl	HIDDEN_JUMPTARGET(exit)
	b	4b

	cfi_startproc
END(__makecontext)

versioned_symbol (libc, __makecontext, makecontext, GLIBC_2_3_4)

#if SHLIB_COMPAT (libc, GLIBC_2_3_3, GLIBC_2_3_4)

	compat_text_section
ENTRY(__novec_makecontext)
	/* Set up the first 7 args to the function in its registers */
	addi	r11,r3,_UC_REG_SPACE
	stw	r11,_UC_REGS_PTR(r3)
	stw	r6,_UC_GREGS+(PT_R3*4)(r11)
	stw	r7,_UC_GREGS+(PT_R4*4)(r11)
	stw	r8,_UC_GREGS+(PT_R5*4)(r11)
	stw	r9,_UC_GREGS+(PT_R6*4)(r11)
	stw	r10,_UC_GREGS+(PT_R7*4)(r11)
	lwz	r8,8(r1)
	lwz	r9,12(r1)
	stw	r8,_UC_GREGS+(PT_R8*4)(r11)
	stw	r9,_UC_GREGS+(PT_R9*4)(r11)

	/* Set the NIP to the start of the function */
	stw	r4,_UC_GREGS+(PT_NIP*4)(r11)

	/* Set the function's r31 to ucp->uc_link for the exitcode below. */
	lwz	r7,_UC_LINK(r3)
	stw	r7,_UC_GREGS+(PT_R31*4)(r11)

	/* Set the function's LR to point to the exitcode below. */
#ifdef PIC
	mflr	r0
	cfi_register(lr,r0)
	/* Use this conditional form of branch and link to avoid destroying
	   the cpu link stack used to predict blr return addresses.  */
	bcl	20,31,1f
1:	mflr	r6
	addi	r6,r6,L(novec_exitcode)-1b
	mtlr	r0
	cfi_same_value (lr)
#else
	lis	r6,L(novec_exitcode)@ha
	addi	r6,r6,L(novec_exitcode)@l
#endif
	stw	r6,_UC_GREGS+(PT_LNK*4)(r11)

	/*
	 * Set up the stack frame for the function.
	 * If we have more than 5 args to the function (8 args to makecontext),
	 * there will be some arguments on the stack which have to end up
	 * in registers.  If there are more than 8 args to the function,
	 * we have to copy (argc - 8) args from our stack to the functions'
	 * stack (and allow space for them in the frame).
	 */
	lwz	r4,_UC_STACK_SP(r3)
	lwz	r8,_UC_STACK_SIZE(r3)
	add	r4,r4,r8
	rlwinm	r4,r4,0,0,27	/* round down to 16-byte boundary */
	addi	r7,r4,-16	/* stack frame for fn's caller */
	cmpwi	r5,8
	blt	2f		/* less than 8 args is easy */
	lwz	r10,16(r1)
	stw	r10,_UC_GREGS+(PT_R10*4)(r11)
	beq	2f		/* if exactly 8 args */
	subi	r9,r5,3
	subi	r5,r5,8
	rlwinm	r9,r9,2,0,27
	subf	r7,r9,r4
	mtctr	r5		/* copy the 9th and following args */
	addi	r6,r1,16
	addi	r8,r7,4
3:	lwzu	r10,4(r6)
	stwu	r10,4(r8)
	bdnz	3b
2:	stw	r7,_UC_GREGS+(PT_R1*4)(r11)
	li	r6,0
	stw	r6,0(r7)

	blr

	cfi_endproc
	nop
/*
 * If the function returns, it comes here.  We put ucp->uc_link in
 * r31, which is a callee-saved register.  We have to continue with
 * the context that r31 points to, or exit if it is 0.
 */
L(novec_exitcode):
	mr.	r3,r31
	beq	4f
	bl	__novec_setcontext@local
4:	bl	HIDDEN_JUMPTARGET(exit)
	b	4b

	cfi_startproc
END(__novec_makecontext)
	.previous

compat_symbol (libc, __novec_makecontext, makecontext, GLIBC_2_3_3)
#endif

#if SHLIB_COMPAT (libc, GLIBC_2_1, GLIBC_2_3_3)

#define _ERRNO_H	1
#include <bits/errno.h>

	compat_text_section
ENTRY (__makecontext_stub)
	li	r3,ENOSYS
	b	__syscall_error@local
END (__makecontext_stub)
	.previous

compat_symbol (libc, __makecontext_stub, makecontext, GLIBC_2_1)

#endif
