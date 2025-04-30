/* Compute the hypothenuse of X and Y.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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
#include <i386-math-asm.h>
#include <libm-alias-finite.h>

	.text
ENTRY(__ieee754_hypotf)
	flds	4(%esp)		// x
	fxam
	fnstsw
	flds	8(%esp)		// y : x
	movb	%ah, %ch
	fxam
	fnstsw
	movb	%ah, %al
	orb	%ch, %ah
	sahf
	jc	1f
	fmul	%st(0)		// y * y : x
	fxch			// x : y * y
	fmul	%st(0)		// x * x : y * y
	faddp			// x * x + y * y
	fsqrt
	FLT_NARROW_EVAL
2:	ret

	// We have to test whether any of the parameters is Inf.
	// In this case the result is infinity.
1:	andb	$0x45, %al
	cmpb	$5, %al
	je	3f		// jump if y is Inf
	andb	$0x45, %ch
	cmpb	$5, %ch
	jne	4f		// jump if x is not Inf
	fxch
3:	fstp	%st(1)
	fabs
	jmp	2b

4:	testb	$1, %al
	jnz	5f		// y is NaN
	fxch
5:	fstp	%st(1)
	jmp	2b

END(__ieee754_hypotf)
libm_alias_finite (__ieee754_hypotf, __hypotf)
