/* Helper macros for x86_64 libm functions.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#ifndef _X86_64_MATH_ASM_H
#define _X86_64_MATH_ASM_H 1

/* Define constants for the minimum value of a floating-point
   type.  */
#define DEFINE_LDBL_MIN					\
	.section .rodata.cst16,"aM",@progbits,16;	\
	.p2align 4;					\
	.type ldbl_min,@object;				\
ldbl_min:						\
	.byte 0, 0, 0, 0, 0, 0, 0, 0x80, 0x1, 0;	\
	.byte 0, 0, 0, 0, 0, 0;				\
	.size ldbl_min, .-ldbl_min;

/* Force an underflow exception if the given value (nonnegative or
   NaN) is subnormal.  The relevant constant for the minimum of the
   type must have been defined, the MO macro must have been defined
   for access to memory operands, and, if PIC, the PIC register must
   have been loaded.  */
#define LDBL_CHECK_FORCE_UFLOW_NONNEG_NAN	\
	fldt	MO(ldbl_min);			\
	fld	%st(1);				\
	fucomip	%st(1), %st(0);			\
	fstp	%st(0);				\
	jnc 6464f;				\
	fld	%st(0);				\
	fmul	%st(0);				\
	fstp	%st(0);				\
6464:

/* Likewise, but the argument is not a NaN.  */
#define LDBL_CHECK_FORCE_UFLOW_NONNAN		\
	fldt	MO(ldbl_min);			\
	fld	%st(1);				\
	fabs;					\
	fcomip	%st(1), %st(0);			\
	fstp	%st(0);				\
	jnc 6464f;				\
	fld	%st(0);				\
	fmul	%st(0);				\
	fstp	%st(0);				\
6464:

/* Likewise, but the argument is nonnegative and not a NaN.  */
#define LDBL_CHECK_FORCE_UFLOW_NONNEG		\
	fldt	MO(ldbl_min);			\
	fld	%st(1);				\
	fcomip	%st(1), %st(0);			\
	fstp	%st(0);				\
	jnc 6464f;				\
	fld	%st(0);				\
	fmul	%st(0);				\
	fstp	%st(0);				\
6464:

#endif /* x86_64-math-asm.h.  */
