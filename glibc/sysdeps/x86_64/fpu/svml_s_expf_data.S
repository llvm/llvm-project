/* Data for function expf.
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

#include "svml_s_expf_data.h"

	.section .rodata, "a"
	.align 64

/* Data table for vector implementations of function expf.
   The table may contain polynomial, reduction, lookup coefficients and
   other coefficients obtained through different methods of research and
   experimental work.  */

	.globl __svml_sexp_data
__svml_sexp_data:

/* Range reduction coefficients:
 * log(2) inverted */
float_vector __sInvLn2 0x3fb8aa3b

/* right shifter constant */
float_vector __sShifter 0x4b400000

/* log(2) high part */
float_vector __sLn2hi 0x3f317200

/* log(2) low part */
float_vector __sLn2lo 0x35bfbe8e

/* bias */
float_vector __iBias 0x0000007f

/* Polynomial coefficients:
 * Here we approximate 2^x on [-0.5, 0.5] */
float_vector __sPC0 0x3f800000
float_vector __sPC1 0x3f7ffffe
float_vector __sPC2 0x3effff34
float_vector __sPC3 0x3e2aacac
float_vector __sPC4 0x3d2b8392
float_vector __sPC5 0x3c07d9fe

/* absolute value mask */
float_vector __iAbsMask 0x7fffffff

/* working domain range */
float_vector __iDomainRange 0x42aeac4f
	.type	__svml_sexp_data,@object
	.size __svml_sexp_data,.-__svml_sexp_data
