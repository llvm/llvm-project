/* Copy a sign bit between floating-point values.
   IBM extended format long double version.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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

ENTRY(__copysignl)
/* long double [f1,f2] copysign (long double [f1,f2] x, long double [f3,f4] y);
   copysign(x,y) returns a value with the magnitude of x and
   with the sign bit of y.  */
#ifdef _ARCH_PPCGR
	/* fsel available.  */
	stwu	r1,-16(r1)
	cfi_adjust_cfa_offset (16)
	stfd	fp3,8(r1)
	fmr	fp0,fp1
	fabs	fp1,fp1
	lwz	r3,8+HIWORD(r1)
	cmpwi	cr6,r3,0
	addi	r1,r1,16
	cfi_adjust_cfa_offset (-16)
	fneg	fp3,fp2
	fsel	fp2,fp0,fp2,fp3
	bgelr	cr6
	fneg	fp1,fp1
	fneg	fp2,fp2
	blr
#else
	stwu	r1,-32(r1)
	cfi_adjust_cfa_offset (32)
	stfd	fp3,8(r1)
	stfd	fp1,16(r1)
	lwz	r3,8+HIWORD(r1)
	lwz	r4,16+HIWORD(r1)
	xor	r3,r3,r4
	cmpwi	cr6,r3,0
	addi	r1,r1,32
	cfi_adjust_cfa_offset (-32)
	bgelr	cr6
	fneg	fp1,fp1
	fneg	fp2,fp2
	blr
#endif
END (__copysignl)

#if IS_IN (libm)
long_double_symbol (libm, __copysignl, copysignl)
#else
long_double_symbol (libc, __copysignl, copysignl)
#endif
