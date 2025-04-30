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

ENTRY_TOCLESS (__fabsl)
/* long double [f1,f2] fabs (long double [f1,f2] x);
   fabs(x,y) returns a value with the magnitude of x and
   with the sign bit of y.  */
	fmr	fp0,fp1
	fabs	fp1,fp1
	fneg	fp3,fp2
	fsel	fp2,fp0,fp2,fp3
	blr
END (__fabsl)

long_double_symbol (libm, __fabsl, fabsl)
