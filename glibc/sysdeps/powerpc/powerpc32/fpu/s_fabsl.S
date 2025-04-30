/* Copy a sign bit between floating-point values.
   IBM extended format long double version.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>
#include <math_ldbl_opt.h>

ENTRY(__fabsl)
/* long double [f1,f2] fabs (long double [f1,f2] x);
   fabs(x,y) returns a value with the magnitude of x and
   with the sign bit of y.  */
#ifdef _ARCH_PPCGR
	/* fsel available.  */
	fmr	fp0,fp1
#else
	/* Use integer operations to test sign of high part to avoid
	   exceptions on sNaNs.  */
	stwu	r1,-16(r1)
	cfi_adjust_cfa_offset (16)
	stfd	fp1,8(r1)
#endif
	fabs	fp1,fp1
#ifdef _ARCH_PPCGR
	fneg	fp3,fp2
	fsel	fp2,fp0,fp2,fp3
#else
	lwz	r3,8+HIWORD(r1)
	cmpwi	cr6,r3,0
	addi	r1,r1,16
	cfi_adjust_cfa_offset (-16)
	bgelr	cr6
	fneg	fp2,fp2
#endif
	blr
END (__fabsl)

long_double_symbol (libm, __fabsl, fabsl)
