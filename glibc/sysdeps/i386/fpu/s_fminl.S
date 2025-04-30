/* Compute minimum of two numbers, regarding NaN as missing argument.
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

#include <libm-alias-ldouble.h>
#include <sysdep.h>

	.text
ENTRY(__fminl)
	fldt	16(%esp)	// y
	fxam
	fnstsw
	fldt	4(%esp)		// y : x

	andb	$0x45, %ah
	cmpb	$0x01, %ah
	je	2f		// y == NaN

	fxam
	fnstsw
	andb	$0x45, %ah
	cmpb	$0x01, %ah
	je	3f		// x == NaN

	fucom	%st(1)
	fnstsw
	sahf
	jc	1f

	fxch	%st(1)
1:	fstp	%st(1)

	ret

2:	// st(1) is a NaN; st(0) may or may not be.
	fxam
	fnstsw
	andb	$0x45, %ah
	cmpb	$0x01, %ah
	je	4f
	// st(1) is a NaN; st(0) is not.  Test if st(1) is signaling.
	testb	$0x40, 23(%esp)
	jz	4f
	fstp	%st(1)
	ret

3:	// st(0) is a NaN; st(1) is not.  Test if st(0) is signaling.
	testb	$0x40, 11(%esp)
	jz	4f
	fstp	%st(0)
	ret

4:	// Both arguments are NaNs, or one is a signaling NaN.
	faddp
	ret
END(__fminl)
libm_alias_ldouble (__fmin, fmin)
