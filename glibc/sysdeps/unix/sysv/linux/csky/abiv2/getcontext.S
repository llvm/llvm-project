/* Save current context.  C-SKY ABIV2 version.
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

#include "ucontext_i.h"

/* int getcontext (ucontext_t *ucp) */

ENTRY (__getcontext)
	/* No need to save r2-r7 or a0-a3.  */
	mov	t0, a0

	addi	a0, MCONTEXT_CSKY_R4

	stm	r4-r13, (a0)			/* Save r4-r13.  */
	stw	sp, (t0, MCONTEXT_CSKY_SP)	/* Save sp.  */
	stw	r15, (t0, MCONTEXT_CSKY_LR)	/* Save lr.  */
	stw	r15, (t0, MCONTEXT_CSKY_PC)	/* Return to PC.  */
	addi	a0, t0, MCONTEXT_CSKY_R16
	stm	r16-r31, (a0)			/* Save r16-r31.  */
	movi	a0, 0
	stw	a0, (t0, MCONTEXT_CSKY_A0)	/* Return zero.  */

	subi	sp, 8
	stw	t0, (sp, 0)			/* Save t0 after "save sp".  */

	/* __sigprocmask (SIG_BLOCK, NULL, &(ucontext->uc_sigmask)) */
	movi	a0, SIG_BLOCK
	movi	a1, 0
	addi	a2, t0, UCONTEXT_SIGMASK

	/* Do sigprocmask syscall.  */
#ifdef __PIC__
	subi	sp, 8
	stw	gb, (sp, 0)
	grs	gb, .Lgetpc
.Lgetpc:
	lrw	a3, .Lgetpc@GOTPC
	addu	gb, gb, a3
	lrw	a3, __sigprocmask@PLT
	ldr.w	a3, (gb, a3 << 0)
	jsr	a3
	ldw	gb, (sp, 0)
	addi	sp, 8
#else
	jsri	__sigprocmask
#endif

	ldw	t0, (sp, 0)
	addi	sp, 8
	/* Restore r15 for sigprocmask changes.  */
	ldw	r15, (t0, MCONTEXT_CSKY_LR)
	movi	a0, 0				/* Return 0.  */
	jmp	r15
END (__getcontext)

weak_alias (__getcontext, getcontext)
