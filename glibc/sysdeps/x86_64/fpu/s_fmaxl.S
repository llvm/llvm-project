/* Compute maximum of two numbers, regarding NaN as missing argument.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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
#include <libm-alias-ldouble.h>

	.text
ENTRY(__fmaxl)
	fldt	8(%rsp)		// x
	fldt	24(%rsp)	// x : y

	fucomi	%st(1), %st
	jp	2f
	fcmovb	%st(1), %st

	fstp	%st(1)

	ret

2:	// Unordered.
	fucomi	%st(0), %st
	jp	3f
	// st(1) is a NaN; st(0) is not.  Test if st(1) is signaling.
	testb	$0x40, 15(%rsp)
	jz	4f
	fstp	%st(1)
	ret

3:	// st(0) is a NaN; st(1) may or may not be.
	fxch
	fucomi	%st(0), %st
	jp	4f
	// st(1) is a NaN; st(0) is not.  Test if st(1) is signaling.
	testb	$0x40, 31(%rsp)
	jz	4f
	fstp	%st(1)
	ret

4:	// Both arguments are NaNs, or one is a signaling NaN.
	faddp
	ret
END(__fmaxl)
libm_alias_ldouble (__fmax, fmax)
