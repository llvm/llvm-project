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

	.set noat

	.align 4
	.globl div
	.ent div
div:
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

	beq	$18, $divbyzero
	excb
	mf_fpcr	$f10

	_ITOFT2	$17, $f0, 0, $18, $f1, 8

	cvtqt	$f0, $f0
	cvtqt	$f1, $f1
	divt/c	$f0, $f1, $f0
	cvttq/c	$f0, $f0
	excb
	mt_fpcr	$f10
	_FTOIT	$f0, $0, 0

	mull	$0, $18, $1
	subl	$17, $1, $1

	stl	$0, 0(a0)
	stl	$1, 4(a0)
	mov	a0, v0

#if FRAME > 0
	lda	sp, FRAME(sp)
#endif
	ret

$divbyzero:
	mov	a0, v0
	lda	a0, GEN_INTDIV
	call_pal PAL_gentrap
	stl	zero, 0(v0)
	stl	zero, 4(v0)

#if FRAME > 0
	lda	sp, FRAME(sp)
#endif
	ret

	.end div
