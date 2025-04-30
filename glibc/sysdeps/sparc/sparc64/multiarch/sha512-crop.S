/* SHA512 using sparc crypto opcodes.
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

#define SHA512		\
	.word	0x81b02860;

	.text
	.align	32
ENTRY(__sha512_process_block_crop)
	/* %o0=buffer, %o1=len, %o2=CTX */
	ldx	[%o2 + 0x48], %g1
	add	%g1, %o1, %o4
	stx	%o4, [%o2 + 0x48]
	cmp	%o4, %g1
	bgeu,pt	%xcc, 1f
	 nop
	ldx	[%o2 + 0x40], %g1
	add	%g1, 1, %g1
	stx	%g1, [%o2 + 0x40]

1:	ldd	[%o2 + 0x00], %f0
	ldd	[%o2 + 0x08], %f2
	ldd	[%o2 + 0x10], %f4
	ldd	[%o2 + 0x18], %f6
	ldd	[%o2 + 0x20], %f8
	ldd	[%o2 + 0x28], %f10
	andcc	%o1, 0x7, %g0
	ldd	[%o2 + 0x30], %f12
	bne,pn	%xcc, 10f
	 ldd	[%o2 + 0x38], %f14

1:
	ldd	[%o0 + 0x00], %f16
	ldd	[%o0 + 0x08], %f18
	ldd	[%o0 + 0x10], %f20
	ldd	[%o0 + 0x18], %f22
	ldd	[%o0 + 0x20], %f24
	ldd	[%o0 + 0x28], %f26
	ldd	[%o0 + 0x30], %f28
	ldd	[%o0 + 0x38], %f30
	ldd	[%o0 + 0x40], %f32
	ldd	[%o0 + 0x48], %f34
	ldd	[%o0 + 0x50], %f36
	ldd	[%o0 + 0x58], %f38
	ldd	[%o0 + 0x60], %f40
	ldd	[%o0 + 0x68], %f42
	ldd	[%o0 + 0x70], %f44
	ldd	[%o0 + 0x78], %f46

	SHA512

	subcc	%o1, 0x80, %o1
	bne,pt	%xcc, 1b
	 add	%o0, 0x80, %o0

5:
	std	%f0, [%o2 + 0x00]
	std	%f2, [%o2 + 0x08]
	std	%f4, [%o2 + 0x10]
	std	%f6, [%o2 + 0x18]
	std	%f8, [%o2 + 0x20]
	std	%f10, [%o2 + 0x28]
	std	%f12, [%o2 + 0x30]
	retl
	 std	%f14, [%o2 + 0x38]
10:
	alignaddr %o0, %g0, %o0

	ldd	[%o0 + 0x00], %f18
1:
	ldd	[%o0 + 0x08], %f20
	ldd	[%o0 + 0x10], %f22
	ldd	[%o0 + 0x18], %f24
	ldd	[%o0 + 0x20], %f26
	ldd	[%o0 + 0x28], %f28
	ldd	[%o0 + 0x30], %f30
	ldd	[%o0 + 0x38], %f32
	ldd	[%o0 + 0x40], %f34
	ldd	[%o0 + 0x48], %f36
	ldd	[%o0 + 0x50], %f38
	ldd	[%o0 + 0x58], %f40
	ldd	[%o0 + 0x60], %f42
	ldd	[%o0 + 0x68], %f44
	ldd	[%o0 + 0x70], %f46
	ldd	[%o0 + 0x78], %f48
	ldd	[%o0 + 0x80], %f50

	faligndata %f18, %f20, %f16
	faligndata %f20, %f22, %f18
	faligndata %f22, %f24, %f20
	faligndata %f24, %f26, %f22
	faligndata %f26, %f28, %f24
	faligndata %f28, %f30, %f26
	faligndata %f30, %f32, %f28
	faligndata %f32, %f34, %f30
	faligndata %f34, %f36, %f32
	faligndata %f36, %f38, %f34
	faligndata %f38, %f40, %f36
	faligndata %f40, %f42, %f38
	faligndata %f42, %f44, %f40
	faligndata %f44, %f46, %f42
	faligndata %f46, %f48, %f44
	faligndata %f48, %f50, %f46

	SHA512

	subcc	%o1, 0x80, %o1
	fsrc2	%f50, %f18
	bne,pt	%xcc, 1b
	 add	%o0, 0x80, %o0

	ba,a,pt	%xcc, 5b
END(__sha512_process_block_crop)
