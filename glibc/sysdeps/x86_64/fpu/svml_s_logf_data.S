/* Data for vector function logf.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#include "svml_s_logf_data.h"

	.section .rodata, "a"
	.align 64

/* Data table for vector implementations of function logf.
   The table may contain polynomial, reduction, lookup coefficients and
   other coefficients obtained through different methods of research and
   experimental work.  */

	.globl __svml_slog_data
__svml_slog_data:

/* Polynomial sPoly[] coefficients:
 * -5.0000000000000000000000000e-01 */
float_vector _sPoly_1 0xbf000000

/* 3.3336564898490905761718750e-01 */
float_vector _sPoly_2 0x3eaaaee7

/* -2.5004664063453674316406250e-01 */
float_vector _sPoly_3 0xbe80061d

/* 1.9822503626346588134765625e-01 */
float_vector _sPoly_4 0x3e4afb81

/* -1.6462457180023193359375000e-01 */
float_vector _sPoly_5 0xbe289358

/* 1.6964881122112274169921875e-01 */
float_vector _sPoly_6 0x3e2db86b

/* -1.5177205204963684082031250e-01 */
float_vector _sPoly_7 0xbe1b6a22

/* Constant for work range check: Delta 80000000-7f800000 */
float_vector _iHiDelta 0x00800000

/* Constant for work range check: 00800000 + Delta */
float_vector _iLoRange 0x01000000

/* Mantissa break point  SP 2/3 */
float_vector _iBrkValue 0x3f2aaaab

/* SP significand mask */
float_vector _iOffExpoMask 0x007fffff

/* 1.0f */
float_vector _sOne 0x3f800000

/* SP log(2) */
float_vector _sLn2 0x3f317218

/* SP infinity, +/- */
/* clang integrated assembler doesn't think subtract yields an absolute, skip.  */
#if !defined(__clang__)
.if .-__svml_slog_data != _sInfs
.err
.endif
#endif
	.long	0x7f800000
	.long	0xff800000
	.rept	56
	.byte	0
	.endr

/* SP one, +/- */
/* clang integrated assembler doesn't think subtract yields an absolute, skip.  */
#if !defined(__clang__)
.if .-__svml_slog_data != _sOnes
.err
.endif
#endif
	.long	0x3f800000
	.long	0xbf800000
	.rept	56
	.byte	0
	.endr

/* SP zero +/- */
/* clang integrated assembler doesn't think subtract yields an absolute, skip.  */
#if !defined(__clang__)
.if .-__svml_slog_data != _sZeros
.err
.endif
#endif
	.long	0x00000000
	.long	0x80000000
	.rept	56
	.byte	0
	.endr
	.type	__svml_slog_data,@object
	.size __svml_slog_data,.-__svml_slog_data
