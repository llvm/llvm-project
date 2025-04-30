/* Round float to int floating-point values without generating
   an inexact exception, sparc64 version.

   Copyright (C) 2013-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David S. Miller <davem@davemloft.net>, 2013.

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

	/* We pop constants into the FPU registers using the incoming
	   argument stack slots, since this avoid having to use any PIC
	   references.  We also thus avoid having to allocate a register
	   window.

	   VIS instructions are used to facilitate the formation of
	   easier constants, and the propagation of the sign bit.  */

#define TWO_TWENTYTHREE	0x4b000000		/* 2**23 */

#define ZERO		%f10			/* 0.0 */
#define SIGN_BIT	%f12			/* -0.0 */

ENTRY (__nearbyintf)
	fcmps	%fcc3, %f1, %f1			/* Check for sNaN */
	stx	%fsr, [%sp + STACK_BIAS + 144]
	sethi	%hi(0xf8003e0), %o5
	sethi	%hi(TWO_TWENTYTHREE), %o2
	ldx	[%sp + STACK_BIAS + 144], %o4
	or	%o5, %lo(0xf8003e0), %o5
	fzeros	ZERO
	andn	%o4, %o5, %o4
	fnegs	ZERO, SIGN_BIT
	st	%o2, [%sp + STACK_BIAS + 128]
	stx	%o4, [%sp + STACK_BIAS + 136]
	ldx	[%sp + STACK_BIAS + 136], %fsr
	fabss	%f1, %f14
	ld	[%sp + STACK_BIAS + 128], %f16
	fcmps	%fcc3, %f14, %f16
	fmovsuge %fcc3, ZERO, %f16
	fands	%f1, SIGN_BIT, SIGN_BIT
	fors	%f16, SIGN_BIT, %f16
	fadds	%f1, %f16, %f5
	fsubs	%f5, %f16, %f0
	fabss	%f0, %f0
	fors	%f0, SIGN_BIT, %f0
	retl
	 ldx	[%sp + STACK_BIAS + 144], %fsr
END (__nearbyintf)
libm_alias_float (__nearbyint, nearbyint)
