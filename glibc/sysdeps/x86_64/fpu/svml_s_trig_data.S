/* Data for function cosf.
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

#include "svml_s_trig_data.h"

	.section .rodata, "a"
	.align 64

/* Data table for vector implementations of function cosf.
   The table may contain polynomial, reduction, lookup coefficients
   and other macro_names obtained through different methods
   of research and experimental work.  */

	.globl __svml_s_trig_data
__svml_s_trig_data:

/* General purpose constants:
   absolute value mask */
float_vector __sAbsMask 0x7fffffff

/* threshold for out-of-range values */
float_vector __sRangeReductionVal 0x461c4000

/* +INF */
float_vector __sRangeVal 0x7f800000

/* High Accuracy version polynomial coefficients:
   S1 = -1.66666666664728165763e-01 */
float_vector __sS1 0xbe2aaaab

/* S2 = 8.33329173045453069014e-03 */
float_vector __sS2 0x3c08885c

/* C1 = -5.00000000000000000000e-01 */
float_vector __sC1 0xbf000000

/* C2 = 4.16638942914469202550e-02 */
float_vector __sC2 0x3d2aaa7c

/* Range reduction PI-based constants:
   PI high part  */
float_vector __sPI1 0x40490000

/* PI mid part 1 */
float_vector __sPI2 0x3a7da000

/* PI mid part 2 */
float_vector __sPI3 0x34222000

/* PI low part */
float_vector __sPI4 0x2cb4611a

/* PI1, PI2, and PI3 when FMA is available
   PI high part (when FMA available) */
float_vector __sPI1_FMA 0x40490fdb

/* PI mid part  (when FMA available) */
float_vector __sPI2_FMA 0xb3bbbd2e

/* PI low part  (when FMA available) */
float_vector __sPI3_FMA 0xa7772ced

/* Polynomial constants for work w/o FMA, relative error ~ 2^(-26.625) */
float_vector __sA3 0xbe2aaaa6
float_vector __sA5 0x3c08876a
float_vector __sA7 0xb94fb7ff
float_vector __sA9 0x362edef8

/* Polynomial constants, work with FMA, relative error ~ 2^(-26.417) */
float_vector __sA5_FMA 0x3c088768
float_vector __sA7_FMA 0xb94fb6cf
float_vector __sA9_FMA 0x362ec335

/* 1/PI */
float_vector __sInvPI 0x3ea2f983

/* right-shifter constant */
float_vector __sRShifter 0x4b400000

/* PI/2 */
float_vector __sHalfPI 0x3fc90fdb

/* 1/2 */
float_vector __sOneHalf 0x3f000000

/* high accuracy table index mask */
float_vector __iIndexMask 0x000000ff

/* 2^(k-1) */
float_vector __i2pK_1 0x00000040

/* sign field mask */
float_vector __sSignMask 0x80000000

	.type	__svml_s_trig_data,@object
	.size __svml_s_trig_data,.-__svml_s_trig_data
