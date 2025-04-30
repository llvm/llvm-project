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


/* 64-bit unsigned long remainder.  These are not normal C functions.  Argument
   registers are t10 and t11, the result goes in t12.  Only t12 and AT may be
   clobbered.

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
	.globl	__remqu
	.type	__remqu, @funcnoplt
	.usepv	__remqu, no

	cfi_startproc
	cfi_return_column (RA)
__remqu:
	lda	sp, -FRAME(sp)
	cfi_def_cfa_offset (FRAME)
	CALL_MCOUNT

	/* Get the fp divide insn issued as quickly as possible.  After
	   that's done, we have at least 22 cycles until its results are
	   ready -- all the time in the world to figure out how we're
	   going to use the results.  */
	subq	Y, 1, AT
	and	Y, AT, AT
	beq	AT, $powerof2

	stt	$f0, 0(sp)
	excb
	stt	$f1, 8(sp)
	stt	$f3, 48(sp)
	cfi_rel_offset ($f0, 0)
	cfi_rel_offset ($f1, 8)
	cfi_rel_offset ($f3, 48)
	mf_fpcr	$f3

	_ITOFT2	X, $f0, 16, Y, $f1, 24
	cvtqt	$f0, $f0
	cvtqt	$f1, $f1

	blt	X, $x_is_neg
	divt/c	$f0, $f1, $f0

	/* Check to see if Y was mis-converted as signed value.  */
	ldt	$f1, 8(sp)
	blt	Y, $y_is_neg

	/* Check to see if X fit in the double as an exact value.  */
	srl	X, 53, AT
	bne	AT, $x_big

	/* If we get here, we're expecting exact results from the division.
	   Do nothing else besides convert, compute remainder, clean up.  */
	cvttq/c	$f0, $f0
	excb
	mt_fpcr	$f3
	_FTOIT	$f0, AT, 16

	mulq	AT, Y, AT
	ldt	$f0, 0(sp)
	ldt	$f3, 48(sp)
	lda	sp, FRAME(sp)
	cfi_remember_state
	cfi_restore ($f0)
	cfi_restore ($f1)
	cfi_restore ($f3)
	cfi_def_cfa_offset (0)

	.align	4
	subq	X, AT, RV
	ret	$31, (RA), 1

	.align	4
	cfi_restore_state
$x_is_neg:
	/* If we get here, X is so big that bit 63 is set, which made the
	   conversion come out negative.  Fix it up lest we not even get
	   a good estimate.  */
	ldah	AT, 0x5f80		/* 2**64 as float.  */
	stt	$f2, 24(sp)
	cfi_rel_offset ($f2, 24)
	_ITOFS	AT, $f2, 16

	.align	4
	addt	$f0, $f2, $f0
	unop
	divt/c	$f0, $f1, $f0
	unop

	/* Ok, we've now the divide issued.  Continue with other checks.  */
	ldt	$f1, 8(sp)
	unop
	ldt	$f2, 24(sp)
	blt	Y, $y_is_neg
	cfi_restore ($f1)
	cfi_restore ($f2)
	cfi_remember_state	/* for y_is_neg */

	.align	4
$x_big:
	/* If we get here, X is large enough that we don't expect exact
	   results, and neither X nor Y got mis-translated for the fp
	   division.  Our task is to take the fp result, figure out how
	   far it's off from the correct result and compute a fixup.  */
	stq	t0, 16(sp)
	stq	t1, 24(sp)
	stq	t2, 32(sp)
	stq	t3, 40(sp)
	cfi_rel_offset (t0, 16)
	cfi_rel_offset (t1, 24)
	cfi_rel_offset (t2, 32)
	cfi_rel_offset (t3, 40)

#define Q	t0		/* quotient */
#define R	RV		/* remainder */
#define SY	t1		/* scaled Y */
#define S	t2		/* scalar */
#define QY	t3		/* Q*Y */

	cvttq/c	$f0, $f0
	_FTOIT	$f0, Q, 8
	mulq	Q, Y, QY

	.align	4
	stq	t4, 8(sp)
	excb
	ldt	$f0, 0(sp)
	mt_fpcr	$f3
	cfi_rel_offset (t4, 8)
	cfi_restore ($f0)

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
	ldq	t4, 8(sp)
	ldq	t0, 16(sp)
	ldq	t1, 24(sp)
	ldq	t2, 32(sp)

	ldq	t3, 40(sp)
	ldt	$f3, 48(sp)
	lda	sp, FRAME(sp)
	cfi_remember_state
	cfi_restore (t0)
	cfi_restore (t1)
	cfi_restore (t2)
	cfi_restore (t3)
	cfi_restore (t4)
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
	cfi_restore_state
$y_is_neg:
	/* If we get here, Y is so big that bit 63 is set.  The results
	   from the divide will be completely wrong.  Fortunately, the
	   quotient must be either 0 or 1, so the remainder must be X
	   or X-Y, so just compute it directly.  */
	cmpule	Y, X, AT
	excb
	mt_fpcr	$f3
	subq	X, Y, RV
	ldt	$f0, 0(sp)
	ldt	$f3, 48(sp)
	cmoveq	AT, X, RV

	lda	sp, FRAME(sp)
	cfi_restore ($f0)
	cfi_restore ($f3)
	cfi_def_cfa_offset (0)
	ret	$31, (RA), 1

	.align	4
	cfi_def_cfa_offset (FRAME)
$powerof2:
	subq	Y, 1, AT
	beq	Y, DIVBYZERO
	and	X, AT, RV
	lda	sp, FRAME(sp)
	cfi_def_cfa_offset (0)
	ret	$31, (RA), 1

	cfi_endproc
	.size	__remqu, .-__remqu

	DO_DIVBYZERO
