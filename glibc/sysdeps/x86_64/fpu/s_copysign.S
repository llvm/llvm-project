/* copy sign, double version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 2002.

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

#include <machine/asm.h>
#include <libm-alias-double.h>

	.section .rodata.cst16,"aM",@progbits,16

	.align ALIGNARG(4)
	.type signmask,@object
signmask:
	.byte 0, 0, 0, 0, 0, 0, 0, 0x80
	.byte 0, 0, 0, 0, 0, 0, 0, 0
	ASM_SIZE_DIRECTIVE(signmask)
	.type othermask,@object
othermask:
	.byte 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f
	.byte 0, 0, 0, 0, 0, 0, 0, 0
	ASM_SIZE_DIRECTIVE(othermask)

#ifdef PIC
#define MO(op) op##(%rip)
#else
#define MO(op) op
#endif

	.text
ENTRY(__copysign)
	andpd MO(othermask),%xmm0
	andpd MO(signmask),%xmm1
	orpd %xmm1,%xmm0
	ret
END (__copysign)

libm_alias_double (__copysign, copysign)
