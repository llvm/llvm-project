/* isnan().  sparc32 v9 version.
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

ENTRY (__isnan)
	sethi	%hi(0x7ff00000), %g1
	sllx	%o0, 33, %o0
	sllx	%g1, 32, %g1
	srlx	%o0, 1, %o0
	or	%o0, %o1, %o0
	sub	%g1, %o0, %o0
	retl
	 srlx	%o0, 63, %o0
END (__isnan)
hidden_def (__isnan)
weak_alias (__isnan, isnan)

#if !IS_IN (libm)
# if LONG_DOUBLE_COMPAT(libc, GLIBC_2_0)
compat_symbol (libc, __isnan, __isnanl, GLIBC_2_0);
compat_symbol (libc, isnan, isnanl, GLIBC_2_0);
# endif
#endif
