/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
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
#include <ucontext-offsets.h>


ENTRY(__makecontext)
	ldgp	$29, 0($27)
#ifdef PROF
	.set noat
	lda     AT, _mcount
	jsr     AT, (AT), _mcount
	.set at
#endif
	.prologue 1

	/* Compute top of stack, including arguments.  */
	ldq	$1, UC_STACK+SS_SP($16)
	ldq	$2, UC_STACK+SS_SIZE($16)
	addq	$1, $2, $8
	subq	$18, 6, $1
	cmovlt	$1, 0, $1
	s8addq	$1, 0, $2
	subq	$8, $2, $8

	/* Copy all parameters.  Switch statement header here.  */
	ldah	$3, $jumptable($29)	!gprelhigh
	cmple	$18, 6, $1
	mov	$18, $2
	cmoveq	$1, 7, $2
	s4addq	$2, $3, $3
	ldl	$4, $jumptable($3)	!gprellow
	addq	$4, $29, $4
	jmp	$31, ($4), $args1

	.section .rodata
	.align	2
$jumptable:
	.gprel32  $args0
	.gprel32  $args1
	.gprel32  $args2
	.gprel32  $args3
	.gprel32  $args4
	.gprel32  $args5
	.gprel32  $args6
	.gprel32  $argsN
	.text

	/* Here we process arguments 7 through N.  This is a straight
	   stack-to-stack copy.  */
	.align	4
$argsN:
	subq	$18, 6, $1
	lda	$2, 0($8)
	lda	$3, 3*8($30)
	.align	4
1:
	ldq	$0, 0($3)
	subq	$1, 1, $1
	lda	$3, 8($3)
	stq	$0, 0($2)
	lda	$2, 8($2)
	bne	$1, 1b

	/* Here we process arguments 6 through 0.  This involves
	   copying into the register save areas of the ucontext.  */
	.align	4
$args6:
	ldq	$0, 2*8($30)
	stq	$0, UC_SIGCTX+SC_REGS+21*8($16)
	unop
	stq	$0, UC_SIGCTX+SC_FPREGS+21*8($16)
$args5:
	ldq	$0, 1*8($30)
	stq	$0, UC_SIGCTX+SC_REGS+20*8($16)
	unop
	stq	$0, UC_SIGCTX+SC_FPREGS+20*8($16)
$args4:
	ldq	$0, 0*8($30)
	stq	$0, UC_SIGCTX+SC_REGS+19*8($16)
	unop
	stq	$0, UC_SIGCTX+SC_FPREGS+19*8($16)
$args3:
	unop
	stq	$21, UC_SIGCTX+SC_REGS+18*8($16)
	unop
	stt	$f21, UC_SIGCTX+SC_FPREGS+18*8($16)
$args2:
	unop
	stq	$20, UC_SIGCTX+SC_REGS+17*8($16)
	unop
	stt	$f20, UC_SIGCTX+SC_FPREGS+17*8($16)
$args1:
	unop
	stq	$19, UC_SIGCTX+SC_REGS+16*8($16)
	unop
	stt	$f19, UC_SIGCTX+SC_FPREGS+16*8($16)
$args0:

	/* Set up the registers ready to invoke __startcontext.
	   We seed $27 with the target function address, and $9
	   with the link from ucp.  */
	ldah	$0, __startcontext($29)		!gprelhigh
	ldq	$1, UC_LINK($16)
	lda	$0, __startcontext($0)		!gprellow
	stq	$17, UC_SIGCTX+SC_REGS+27*8($16)
	stq	$8, UC_SIGCTX+SC_REGS+30*8($16)
	stq	$0, UC_SIGCTX+SC_PC($16)
	stq	$1, UC_SIGCTX+SC_REGS+9*8($16)

	/* No return value from makecontext.  */
	ret

END(__makecontext)
weak_alias (__makecontext, makecontext)

/* This function is where a new makecontext "thread" begins life.
   We have already set up $27 for calling the target function, and
   we've set $9 to the UC_LINK of the parent context.

   If the function returns, we either jump to the linked context
   (if non-null) or exit.  */

	.align	4
	.ent	__startcontext
__startcontext:
	.frame $31, 0, $31, 0
	.prologue 0

	jsr	$26, ($27), 0
	ldgp	$29, 0($26)
	mov	$9, $16
	beq	$9, 1f

#ifdef PIC
	bsr	$26, __setcontext		!samegp
1:	mov	$31, $16
	bsr	$26, HIDDEN_JUMPTARGET(exit)	!samegp
#else
	jsr	$26, __setcontext
	ldgp	$29, 0($26)
1:	mov	$31, $16
	jsr	$26, HIDDEN_JUMPTARGET(exit)
#endif

	halt

	.end __startcontext
