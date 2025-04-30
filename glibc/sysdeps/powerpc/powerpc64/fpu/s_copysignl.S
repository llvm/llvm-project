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

ENTRY_TOCLESS (__copysignl)
/* long double [f1,f2] copysign (long double [f1,f2] x, long double [f3,f4] y);
   copysign(x,y) returns a value with the magnitude of x and
   with the sign bit of y.  */
	stfd	fp3,-16(r1)
	ld	r3,-16(r1)
	cmpdi   r3,0
	blt     L(0)
	fmr	fp0,fp1
	fabs	fp1,fp1
	fneg	fp3,fp2
	fsel	fp2,fp0,fp2,fp3
	blr
L(0):
	fmr	fp0,fp1
	fnabs   fp1,fp1
	fneg	fp3,fp2
	fsel	fp2,fp0,fp3,fp2
	blr
END (__copysignl)

#if IS_IN (libm)
long_double_symbol (libm, __copysignl, copysignl)
#else
long_double_symbol (libc, __copysignl, copysignl)
#endif
