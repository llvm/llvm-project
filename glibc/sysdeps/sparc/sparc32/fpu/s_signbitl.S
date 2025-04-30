/* signbitl().  sparc32 version.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

ENTRY (___signbitl)
	ld	[%o0], %o1
	retl
	 srl	%o1, 31, %o0
END (___signbitl)

#if IS_IN (libm)
long_double_symbol (libm, ___signbitl, __signbitl);
#else
long_double_symbol (libc, ___signbitl, __signbitl);
#endif
