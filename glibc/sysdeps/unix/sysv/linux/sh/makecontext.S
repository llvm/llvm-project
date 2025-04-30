/* Create new context.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>
#include <abort-instr.h>

#include "ucontext_i.h"

/* void __makecontext (ucontext_t *ucp, void (*func)(), int argc, ...);
  __makecontext sets up a stack and registers for context to run a given
  function.  The registers are set up like this:
    r4-r7: parameters 1 to 4
    r8	 : uc_link from ucontext structure
    pc	 : (*func) pointer
    pr	 : address of exitcode
    r15	 : stack pointer for func.  */

	.text
	.align	5
ENTRY(__makecontext)
	mov	#4, r3
	mov.l	@(oSS_SP,r4), r1	/* r1 <- ucb->stack_base */
	mov.l	@(oSS_SIZE,r4), r2	/* r2 <- ucb->stack_size */
	add	r1, r2			/* r2 <- stack_top */
	cmp/gt	r3, r6			/* argc > 4? */
	bf/s	1f
	 shlr2	r2			/* r2 <- stack_top / 4 */
	sub	r6, r2
	add	r3, r2
1:
	shll2	r2		/* r2 <- stack_top - ((argc > 4) ? 4 * (argc - 4) : 0 */
	mov	#oR15, r0
	mov.l	@(oLINK,r4), r1
	mov.l	r2, @(r0,r4)	/* ucp->r15 <- r2 */
	mov.l	r1, @(oR8,r4)	/* ucp->r8 <- ucp->uc_link */
	mov	#oPC, r0
	mov.l	r5, @(r0,r4)	/* ucp->PC <- func */

	cmp/pl	r6		/* argc > 0? */
	bf/s	.L1
	 dt	r6
	mov.l	r7, @(oR4,r4)	/* ucp->r4 <- argv[0] */
	cmp/pl	r6		/* argc > 1? */
	bf/s	.L1
	 dt	r6
	mov.l	@(0,r15), r1
	mov.l	r1, @(oR5,r4)	/* ucp->r5 <- argv[1] */
	cmp/pl	r6		/* argc > 2? */
	bf/s	.L1
	 dt	r6
	mov.l	@(4,r15), r1
	mov.l	r1, @(oR6,r4)	/* ucp->r6 <- argv[2] */
	cmp/pl	r6		/* argc > 3? */
	bf/s	.L1
	 dt	r6
	mov.l	@(8,r15), r1
	mov.l	r1, @(oR7,r4)	/* ucp->r7 <- argv[3] */
	mov	#12,r0
.L0:				/* Save remaining argv[] on the stack.  */
	cmp/pl	r6
	bf/s	.L1
	 dt	r6
	mov.l	@(r0,r15), r1
	mov.l	r1, @r2
	add	#4, r0
	bra	.L0
	 add	#4, r2
.L1:
#ifdef PIC
	mova	.Lexitcode, r0
#else
	mov.l	.L2, r0
#endif
	add	#oPR, r4
	rts
	 mov.l	r0, @r4		/* ucp->pr <- exitcode */
#ifndef PIC
	.align	2
.L2:
	.long	.Lexitcode
#endif
	cfi_endproc

	.align	5
	cfi_startproc
.Lexitcode:
#ifdef PIC
	mov.l	r12, @-r15
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (r12, 0)
	mova	.Lgot, r0
	mov.l	.Lgot, r12
	add	r0, r12
#endif
	tst	r8, r8		/* ucb->uc_link == NULL? */
	bt/s	2f
	 mov	r8, r4		/* r4 <- ucb->uc_link */
	mov.l	.Lsetcontext, r1
	sts.l	pr, @-r15
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (pr, 0)
#ifdef PIC
	bsrf	r1
.LPCS0:
	 nop
#else
	jsr	@r1
	 nop
#endif
        /* Restore to keep CFI/CFA balanced.  */
	lds.l	@r15+, pr
	cfi_adjust_cfa_offset (-4)
	cfi_restore (pr)
	/* If this returns (which can happen if the syscall fails) we'll exit
	   the program with the return error value (-1).  */
	mov	r0, r4

2:
	mov.l	.Lexit, r1
#ifdef PIC
	add	r12, r1
#endif
	jsr	@r1
	 nop
	/* The 'exit' call should never return.  In case it does cause the
	   process to terminate.  */
	ABORT_INSTRUCTION_ASM

	.align	2
#ifdef PIC
.Lgot:
	.long	_GLOBAL_OFFSET_TABLE_
.Lsetcontext:
	.long	__setcontext@PLT-(.LPCS0+2-(.))
.Lexit:
	.long	HIDDEN_JUMPTARGET(exit)@GOTOFF
#else
.Lsetcontext:
	.long	__setcontext
.Lexit:
	.long	HIDDEN_JUMPTARGET(exit)
#endif
PSEUDO_END(__makecontext)

weak_alias (__makecontext, makecontext)
