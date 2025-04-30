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

/* 32-bit signed int divide.  This is not a normal C function.  Argument
   registers are t10 and t11, the result goes in t12.  Only t12 and AT may
   be clobbered.

   The FPU can handle all input values except zero.  Whee!

   The FPCR save/restore is due to the fact that the EV6 _will_ set FPCR_INE
   for cvttq/c even without /sui being set.  It will not, however, properly
   raise the exception, so we don't have to worry about FPCR_INED being clear
   and so dying by SIGFPE.  */

#ifndef EXTEND
#define EXTEND(S,D)	sextl S, D
#endif

	.text
	.align	4
	.globl	__divl
	.type	__divl, @funcnoplt
	.usepv	__divl, no

	cfi_startproc
	cfi_return_column (RA)
__divl:
	lda	sp, -FRAME(sp)
	cfi_def_cfa_offset (FRAME)
	CALL_MCOUNT
	stt	$f0, 0(sp)
	excb
	beq	Y, DIVBYZERO

	stt	$f1, 8(sp)
	stt	$f2, 16(sp)
	cfi_rel_offset ($f0, 0)
	cfi_rel_offset ($f1, 8)
	cfi_rel_offset ($f2, 16)
	mf_fpcr	$f2

	EXTEND	(X, RV)
	EXTEND	(Y, AT)
	_ITOFT2	RV, $f0, 24, AT, $f1, 32
	cvtqt	$f0, $f0
	cvtqt	$f1, $f1
	divt/c	$f0, $f1, $f0
	cvttq/c	$f0, $f0
	excb
	mt_fpcr	$f2
	_FTOIT	$f0, RV, 24

	ldt	$f0, 0(sp)
	ldt	$f1, 8(sp)
	ldt	$f2, 16(sp)
	lda	sp, FRAME(sp)
	cfi_restore ($f0)
	cfi_restore ($f1)
	cfi_restore ($f2)
	cfi_def_cfa_offset (0)
	sextl	RV, RV
	ret	$31, (RA), 1

	cfi_endproc
	.size	__divl, .-__divl

	DO_DIVBYZERO
