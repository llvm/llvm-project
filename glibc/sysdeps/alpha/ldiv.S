/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson <rth@tamu.edu>.

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

#undef FRAME
#ifdef __alpha_fix__
#define FRAME 0
#else
#define FRAME 16
#endif

#undef X
#undef Y
#define X $17
#define Y $18

	.set noat

	.align 4
	.globl ldiv
	.ent ldiv
ldiv:
	.frame sp, FRAME, ra
#if FRAME > 0
	lda	sp, -FRAME(sp)
#endif
#ifdef PROF
	.set	macro
	ldgp	gp, 0(pv)
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.set	nomacro
	.prologue 1
#else
	.prologue 0
#endif

	beq	Y, $divbyzero
	excb
	mf_fpcr	$f10

	_ITOFT2	X, $f0, 0, Y, $f1, 8

	.align	4
	cvtqt	$f0, $f0
	cvtqt	$f1, $f1
	divt/c	$f0, $f1, $f0
	unop

	/* Check to see if X fit in the double as an exact value.  */
	sll	X, (64-53), AT
	sra	AT, (64-53), AT
	cmpeq	X, AT, AT
	beq	AT, $x_big

	/* If we get here, we're expecting exact results from the division.
	   Do nothing else besides convert and clean up.  */
	cvttq/c	$f0, $f0
	excb
	mt_fpcr	$f10
	_FTOIT	$f0, $0, 0

$egress:
	mulq	$0, Y, $1
	subq	X, $1, $1

	stq	$0, 0($16)
	stq	$1, 8($16)
	mov	$16, $0

#if FRAME > 0
	lda	sp, FRAME(sp)
#endif
	ret

	.align	4
$x_big:
	/* If we get here, X is large enough that we don't expect exact
	   results, and neither X nor Y got mis-translated for the fp
	   division.  Our task is to take the fp result, figure out how
	   far it's off from the correct result and compute a fixup.  */

#define Q	v0		/* quotient */
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
$fix_sign_in_ret2:
	mulq	Q, Y, QY
	excb
	mt_fpcr	$f10

	.align	4
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
	negq	Q, t4
	cmovlbs	t5, t4, Q
	br	$egress

	.align	4
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
	/* T5 is true if result should be negative.  */
	xor	X, Y, AT
	cmplt	AT, 0, t5
	cmplt	X, 0, AT
	negq	X, t0

	cmovne	AT, t0, X
	cmplt	Y, 0, AT
	negq	Y, t0

	cmovne	AT, t0, Y
	blbc	t5, $fix_sign_in_ret1

	cvttq/c	$f0, $f0
	_FTOIT	$f0, Q, 8
	.align	3
	negq	Q, Q
	br	$fix_sign_in_ret2

$divbyzero:
	mov	a0, v0
	lda	a0, GEN_INTDIV
	call_pal PAL_gentrap
	stq	zero, 0(v0)
	stq	zero, 8(v0)

#if FRAME > 0
	lda	sp, FRAME(sp)
#endif
	ret

	.end	ldiv

weak_alias (ldiv, lldiv)
weak_alias (ldiv, imaxdiv)
