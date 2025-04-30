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

#include "div_libc.h"


/* 64-bit signed long divide.  These are not normal C functions.  Argument
   registers are t10 and t11, the result goes in t12.  Only t12 and AT may
   be clobbered.

   Theory of operation here is that we can use the FPU divider for virtually
   all operands that we see: all dividend values between -2**53 and 2**53-1
   can be computed directly.  Note that divisor values need not be checked
   against that range because the rounded fp value will be close enough such
   that the quotient is < 1, which will properly be truncated to zero when we
   convert back to integer.

   When the dividend is outside the range for which we can compute exact
   results, we use the fp quotent as an estimate from which we begin refining
   an exact integral value.  This reduces the number of iterations in the
   shift-and-subtract loop significantly.

   The FPCR save/restore is due to the fact that the EV6 _will_ set FPCR_INE
   for cvttq/c even without /sui being set.  It will not, however, properly
   raise the exception, so we don't have to worry about FPCR_INED being clear
   and so dying by SIGFPE.  */

	.text
	.align	4
	.globl	__divq
	.type	__divq, @funcnoplt
	.usepv	__divq, no

	cfi_startproc
	cfi_return_column (RA)
__divq:
	lda	sp, -FRAME(sp)
	cfi_def_cfa_offset (FRAME)
	CALL_MCOUNT

	/* Get the fp divide insn issued as quickly as possible.  After
	   that's done, we have at least 22 cycles until its results are
	   ready -- all the time in the world to figure out how we're
	   going to use the results.  */
	stt	$f0, 0(sp)
	excb
	beq	Y, DIVBYZERO

	stt	$f1, 8(sp)
	stt	$f3, 48(sp)
	cfi_rel_offset ($f0, 0)
	cfi_rel_offset ($f1, 8)
	cfi_rel_offset ($f3, 48)
	mf_fpcr	$f3

	_ITOFT2	X, $f0, 16, Y, $f1, 24
	cvtqt	$f0, $f0
	cvtqt	$f1, $f1
	divt/c	$f0, $f1, $f0

	/* Check to see if X fit in the double as an exact value.  */
	sll	X, (64-53), AT
	ldt	$f1, 8(sp)
	sra	AT, (64-53), AT
	cmpeq	X, AT, AT
	beq	AT, $x_big

	/* If we get here, we're expecting exact results from the division.
	   Do nothing else besides convert and clean up.  */
	cvttq/c	$f0, $f0
	excb
	mt_fpcr	$f3
	_FTOIT	$f0, RV, 16

	ldt	$f0, 0(sp)
	ldt	$f3, 48(sp)
	cfi_restore ($f1)
	cfi_remember_state
	cfi_restore ($f0)
	cfi_restore ($f3)
	cfi_def_cfa_offset (0)
	lda	sp, FRAME(sp)
	ret	$31, (RA), 1

	.align	4
	cfi_restore_state
$x_big:
	/* If we get here, X is large enough that we don't expect exact
	   results, and neither X nor Y got mis-translated for the fp
	   division.  Our task is to take the fp result, figure out how
	   far it's off from the correct result and compute a fixup.  */
	stq	t0, 16(sp)
	stq	t1, 24(sp)
	stq	t2, 32(sp)
	stq	t5, 40(sp)
	cfi_rel_offset (t0, 16)
	cfi_rel_offset (t1, 24)
	cfi_rel_offset (t2, 32)
	cfi_rel_offset (t5, 40)

#define Q	RV		/* quotient */
#define R	t0		/* remainder */
#define SY	t1		/* scaled Y */
#define S	t2		/* scalar */
#define QY	t3		/* Q*Y */

	/* The fixup code below can only handle unsigned values.  */
	or	X, Y, AT
	mov	$31, t5
	blt	AT, $fix_sign_in
$fix_sign_in_ret1:
	cvttq/c	$f0, $f0

	_FTOIT	$f0, Q, 8
	.align	3
$fix_sign_in_ret2:
	ldt	$f0, 0(sp)
	stq	t3, 0(sp)
	cfi_restore ($f0)
	cfi_rel_offset (t3, 0)

	mulq	Q, Y, QY
	excb
	stq	t4, 8(sp)
	mt_fpcr	$f3
	cfi_rel_offset (t4, 8)

	subq	QY, X, R
	mov	Y, SY
	mov	1, S
	bgt	R, $q_high

$q_high_ret:
	subq	X, QY, R
	mov	Y, SY
	mov	1, S
	bgt	R, $q_low

$q_low_ret:
	ldq	t0, 16(sp)
	ldq	t1, 24(sp)
	ldq	t2, 32(sp)
	bne	t5, $fix_sign_out

$fix_sign_out_ret:
	ldq	t3, 0(sp)
	ldq	t4, 8(sp)
	ldq	t5, 40(sp)
	ldt	$f3, 48(sp)
	lda	sp, FRAME(sp)
	cfi_remember_state
	cfi_restore (t0)
	cfi_restore (t1)
	cfi_restore (t2)
	cfi_restore (t3)
	cfi_restore (t4)
	cfi_restore (t5)
	cfi_restore ($f3)
	cfi_def_cfa_offset (0)
	ret	$31, (RA), 1

	.align	4
	cfi_restore_state
	/* The quotient that we computed was too large.  We need to reduce
	   it by S such that Y*S >= R.  Obviously the closer we get to the
	   correct value the better, but overshooting high is ok, as we'll
	   fix that up later.  */
0:
	addq	SY, SY, SY
	addq	S, S, S
$q_high:
	cmpult	SY, R, AT
	bne	AT, 0b

	subq	Q, S, Q
	unop
	subq	QY, SY, QY
	br	$q_high_ret

	.align	4
	/* The quotient that we computed was too small.  Divide Y by the
	   current remainder (R) and add that to the existing quotient (Q).
	   The expectation, of course, is that R is much smaller than X.  */
	/* Begin with a shift-up loop.  Compute S such that Y*S >= R.  We
	   already have a copy of Y in SY and the value 1 in S.  */
0:
	addq	SY, SY, SY
	addq	S, S, S
$q_low:
	cmpult	SY, R, AT
	bne	AT, 0b

	/* Shift-down and subtract loop.  Each iteration compares our scaled
	   Y (SY) with the remainder (R); if SY <= R then X is divisible by
	   Y's scalar (S) so add it to the quotient (Q).  */
2:	addq	Q, S, t3
	srl	S, 1, S
	cmpule	SY, R, AT
	subq	R, SY, t4

	cmovne	AT, t3, Q
	cmovne	AT, t4, R
	srl	SY, 1, SY
	bne	S, 2b

	br	$q_low_ret

	.align	4
$fix_sign_in:
	/* If we got here, then X|Y is negative.  Need to adjust everything
	   such that we're doing unsigned division in the fixup loop.  */
	/* T5 records the changes we had to make:
		bit 0:	set if result should be negative.
		bit 2:	set if X was negated.
		bit 3:	set if Y was negated.
	*/
	xor	X, Y, AT
	cmplt	AT, 0, t5
	cmplt	X, 0, AT
	negq	X, t0

	s4addq	AT, t5, t5
	cmovne	AT, t0, X
	cmplt	Y, 0, AT
	negq	Y, t0

	s8addq	AT, t5, t5
	cmovne	AT, t0, Y
	unop
	blbc	t5, $fix_sign_in_ret1

	cvttq/c	$f0, $f0
	_FTOIT	$f0, Q, 8
	.align	3
	negq	Q, Q
	br	$fix_sign_in_ret2

	.align	4
$fix_sign_out:
	/* Now we get to undo what we did above.  */
	/* ??? Is this really faster than just increasing the size of
	   the stack frame and storing X and Y in memory?  */
	and	t5, 8, AT
	negq	Y, t4
	cmovne	AT, t4, Y

	and	t5, 4, AT
	negq	X, t4
	cmovne	AT, t4, X

	negq	RV, t4
	cmovlbs	t5, t4, RV

	br	$fix_sign_out_ret

	cfi_endproc
	.size	__divq, .-__divq

	DO_DIVBYZERO
