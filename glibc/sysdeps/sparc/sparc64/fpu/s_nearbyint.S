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
#include <libm-alias-double.h>

	/* We pop constants into the FPU registers using the incoming
	   argument stack slots, since this avoid having to use any PIC
	   references.  We also thus avoid having to allocate a register
	   window.

	   VIS instructions are used to facilitate the formation of
	   easier constants, and the propagation of the sign bit.  */

#define TWO_FIFTYTWO	0x43300000		/* 2**52 */

#define ZERO		%f10			/* 0.0 */
#define SIGN_BIT	%f12			/* -0.0 */

ENTRY (__nearbyint)
	fcmpd	%fcc3, %f0, %f0			/* Check for sNaN */
	stx	%fsr, [%sp + STACK_BIAS + 144]
	sethi	%hi(TWO_FIFTYTWO), %o2
	sllx	%o2, 32, %o2
	ldx	[%sp + STACK_BIAS + 144], %o4
	sethi	%hi(0xf8003e0), %o5
	fzero	ZERO
	or	%o5, %lo(0xf8003e0), %o5
	fnegd	ZERO, SIGN_BIT
	andn	%o4, %o5, %o4
	stx	%o2, [%sp + STACK_BIAS + 128]
	stx	%o4, [%sp + STACK_BIAS + 136]
	ldx	[%sp + STACK_BIAS + 136], %fsr
	fabsd	%f0, %f14
	ldd	[%sp + STACK_BIAS + 128], %f16
	fcmpd	%fcc3, %f14, %f16
	fmovduge %fcc3, ZERO, %f16
	fand	%f0, SIGN_BIT, SIGN_BIT
	for	%f16, SIGN_BIT, %f16
	faddd	%f0, %f16, %f6
	fsubd	%f6, %f16, %f0
	fabsd	%f0, %f0
	for	%f0, SIGN_BIT, %f0
	retl
	 ldx	[%sp + STACK_BIAS + 144], %fsr
END (__nearbyint)
libm_alias_double (__nearbyint, nearbyint)
