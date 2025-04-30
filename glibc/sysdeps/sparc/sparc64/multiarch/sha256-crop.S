/* SHA256 using sparc crypto opcodes.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David S. Miller (davem@davemloft.net)

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

#define SHA256		\
	.word	0x81b02840;

	.text
	.align	32
ENTRY(__sha256_process_block_crop)
	/* %o0=buffer, %o1=len, %o2=CTX */
	ldx	[%o2 + 0x20], %g1
	add	%g1, %o1, %g1
	stx	%g1, [%o2 + 0x20]

	ld	[%o2 + 0x00], %f0
	ld	[%o2 + 0x04], %f1
	ld	[%o2 + 0x08], %f2
	ld	[%o2 + 0x0c], %f3
	ld	[%o2 + 0x10], %f4
	ld	[%o2 + 0x14], %f5
	andcc	%o1, 0x7, %g0
	ld	[%o2 + 0x18], %f6
	bne,pn	%xcc, 10f
	 ld	[%o2 + 0x1c], %f7

1:
	ldd	[%o0 + 0x00], %f8
	ldd	[%o0 + 0x08], %f10
	ldd	[%o0 + 0x10], %f12
	ldd	[%o0 + 0x18], %f14
	ldd	[%o0 + 0x20], %f16
	ldd	[%o0 + 0x28], %f18
	ldd	[%o0 + 0x30], %f20
	ldd	[%o0 + 0x38], %f22

	SHA256

	subcc	%o1, 0x40, %o1
	bne,pt	%xcc, 1b
	 add	%o0, 0x40, %o0

5:
	st	%f0, [%o2 + 0x00]
	st	%f1, [%o2 + 0x04]
	st	%f2, [%o2 + 0x08]
	st	%f3, [%o2 + 0x0c]
	st	%f4, [%o2 + 0x10]
	st	%f5, [%o2 + 0x14]
	st	%f6, [%o2 + 0x18]
	retl
	 st	%f7, [%o2 + 0x1c]
10:
	alignaddr %o0, %g0, %o0

	ldd	[%o0 + 0x00], %f10
1:
	ldd	[%o0 + 0x08], %f12
	ldd	[%o0 + 0x10], %f14
	ldd	[%o0 + 0x18], %f16
	ldd	[%o0 + 0x20], %f18
	ldd	[%o0 + 0x28], %f20
	ldd	[%o0 + 0x30], %f22
	ldd	[%o0 + 0x38], %f24
	ldd	[%o0 + 0x40], %f26

	faligndata %f10, %f12, %f8
	faligndata %f12, %f14, %f10
	faligndata %f14, %f16, %f12
	faligndata %f16, %f18, %f14
	faligndata %f18, %f20, %f16
	faligndata %f20, %f22, %f18
	faligndata %f22, %f24, %f20
	faligndata %f24, %f26, %f22

	SHA256

	subcc	%o1, 0x40, %o1
	fsrc2	%f26, %f10
	bne,pt	%xcc, 1b
	 add	%o0, 0x40, %o0

	ba,a,pt	%xcc, 5b
END(__sha256_process_block_crop)
