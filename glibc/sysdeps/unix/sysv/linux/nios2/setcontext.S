/* Set current context.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
#include "ucontext_i.h"

/* int setcontext (const ucontext_t *ucp) */
	.text
ENTRY(__setcontext)
	ldw	r5, UCONTEXT_FLAGS(r4)
	movi	r6, 1
	bne	r5, r6, .Lsigreturn

	mov	r10, r4

	/* Restore signal mask.  */
	/* rt_sigprocmask (SIG_SETMASK, &ucp->uc_sigmask, NULL, _NSIG8) */
	movi	r7, _NSIG8
	addi	r5, r4, UCONTEXT_SIGMASK
	mov	r6, zero
	movi	r4, SIG_SETMASK
	movi	r2, SYS_ify (rt_sigprocmask)
	trap
	bne	r7, zero, SYSCALL_ERROR_LABEL

	/* Restore argument registers, for the makecontext() case.  */
	ldw	r4,   (UCONTEXT_MCONTEXT + 4*4)(r10)
	ldw	r5,   (UCONTEXT_MCONTEXT + 5*4)(r10)
	ldw	r6,   (UCONTEXT_MCONTEXT + 6*4)(r10)
	ldw	r7,   (UCONTEXT_MCONTEXT + 7*4)(r10)

	ldw	r16, (UCONTEXT_MCONTEXT + 16*4)(r10)
	ldw	r17, (UCONTEXT_MCONTEXT + 17*4)(r10)
	ldw	r18, (UCONTEXT_MCONTEXT + 18*4)(r10)
	ldw	r19, (UCONTEXT_MCONTEXT + 19*4)(r10)
	ldw	r20, (UCONTEXT_MCONTEXT + 20*4)(r10)
	ldw	r21, (UCONTEXT_MCONTEXT + 21*4)(r10)
	ldw	r22, (UCONTEXT_MCONTEXT + 22*4)(r10)
	ldw	ra,  (UCONTEXT_MCONTEXT + 24*4)(r10)
	ldw	fp,  (UCONTEXT_MCONTEXT + 25*4)(r10)
	ldw	gp,  (UCONTEXT_MCONTEXT + 26*4)(r10)
	/* Load address to continue execution.  */
	ldw	r3,  (UCONTEXT_MCONTEXT + 28*4)(r10)
	ldw	sp,  (UCONTEXT_MCONTEXT + 29*4)(r10)

	mov	r2, zero
	jmp	r3

.Lsigreturn:
	addi	sp, sp, -RT_SIGFRAME_SIZE
	cfi_adjust_cfa_offset (RT_SIGFRAME_SIZE)

	addi	r2, sp, RT_SIGFRAME_UCONTEXT
	movi	r3, UCONTEXT_SIZE-4
1:
	add	r6, r4, r3
	ldw	r5, 0(r6)
	add	r7, r2, r3
	addi	r3, r3, -4
	stw	r5, 0(r7)
	bgt	r3, zero, 1b

	movi	r2, SYS_ify (rt_sigreturn)
	trap

	addi	sp, sp, RT_SIGFRAME_SIZE
	cfi_adjust_cfa_offset (-RT_SIGFRAME_SIZE)
	br	SYSCALL_ERROR_LABEL

PSEUDO_END (__setcontext)
weak_alias (__setcontext, setcontext)

	/* We add an NOP here to separate between __setcontext/__startcontext.
	   The wanted behavior that happens is: when unwinding from a function
	   called inside a makecontext() context, FDE lookup will use
	   '&__startcontext - 1', then returns NULL for no FDE found,
	   and immediately ends the unwind, in a normal fashion.

	   If this NOP word does not exist, FDE lookup just repeatedly finds
	   __setcontext's FDE in an infinite loop, due to the convention of
	   using 'address - 1' for FDE lookup. Modifiying/deleting the below
	   __startcontext's FDE has no help on this.  */
	nop

ENTRY(__startcontext)
	mov	r4, r16
	bne	r4, zero, __setcontext

	/* If uc_link == zero, call exit.  */
#ifdef PIC
	nextpc	r22
1:	movhi	r8, %hiadj(_gp_got - 1b)
	addi	r8, r8, %lo(_gp_got - 1b)
	add	r22, r22, r8
	ldw	r8, %call(HIDDEN_JUMPTARGET(exit))(r22)
	jmp	r8
#else
	jmpi	exit
#endif
END(__startcontext)
