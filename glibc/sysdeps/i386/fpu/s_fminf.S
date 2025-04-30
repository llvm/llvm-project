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

#include <sysdep.h>
#include <libm-alias-float.h>

	.text
ENTRY(__fminf)
	flds	4(%esp)		// x
	flds	8(%esp)		// x : y

	fxam
	fnstsw
	andb	$0x45, %ah
	cmpb	$0x01, %ah
	je	1f		// y == NaN

	fucom	%st(1)
	fnstsw
	sahf
	jc	2f

1:	fxch	%st(1)
2:	fstp	%st(1)

	ret
END(__fminf)
libm_alias_float (__fmin, fmin)
