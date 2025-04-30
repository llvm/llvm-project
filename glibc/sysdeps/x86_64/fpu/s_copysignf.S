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
#include <libm-alias-float.h>

	.section .rodata

	.align ALIGNARG(4)
	.type mask,@object
mask:
	.byte 0xff, 0xff, 0xff, 0x7f
	ASM_SIZE_DIRECTIVE(mask)

#ifdef PIC
#define MO(op) op##(%rip)
#else
#define MO(op) op
#endif

	.text
ENTRY(__copysignf)
	movss	MO(mask),%xmm3
	andps	%xmm3,%xmm0
	andnps	%xmm1,%xmm3
	orps	%xmm3,%xmm0
	retq
END (__copysignf)

libm_alias_float (__copysign, copysign)
