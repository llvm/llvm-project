/* MD5 using sparc crypto opcodes.
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

#define ASI_PL 0x88

#define MD5		\
	.word	0x81b02800;

	.text
	.align	32
ENTRY(__md5_process_block_crop)
	/* %o0=buffer, %o1=len, %o2=CTX */
	ld	[%o2 + 0x10], %g1
	add	%g1, %o1, %o4
	st	%o4, [%o2 + 0x10]
	clr	%o5
	cmp	%o4, %g1
	movlu	%icc, 1, %o5
#ifdef __arch64__
	srlx	%o1, 32, %o4
	add	%o5, %o4, %o5
#endif
	ld	[%o2 + 0x14], %o4
	add	%o4, %o5, %o4
	st	%o4, [%o2 + 0x14]
	lda	[%o2] ASI_PL, %f0
	add	%o2, 0x4, %g1
	lda	[%g1] ASI_PL, %f1
	add	%o2, 0x8, %g1
	andcc	%o0, 0x7, %g0
	lda	[%g1] ASI_PL, %f2
	add	%o2, 0xc, %g1
	bne,pn	%xcc, 10f
	 lda	[%g1] ASI_PL, %f3

1:
	ldd	[%o0 + 0x00], %f8
	ldd	[%o0 + 0x08], %f10
	ldd	[%o0 + 0x10], %f12
	ldd	[%o0 + 0x18], %f14
	ldd	[%o0 + 0x20], %f16
	ldd	[%o0 + 0x28], %f18
	ldd	[%o0 + 0x30], %f20
	ldd	[%o0 + 0x38], %f22

	MD5

	subcc	%o1, 64, %o1
	bne,pt	%xcc, 1b
	 add	%o0, 0x40, %o0

5:
	sta	%f0, [%o2] ASI_PL
	add	%o2, 0x4, %g1
	sta	%f1, [%g1] ASI_PL
	add	%o2, 0x8, %g1
	sta	%f2, [%g1] ASI_PL
	add	%o2, 0xc, %g1
	retl
	 sta	%f3, [%g1] ASI_PL
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

	MD5

	subcc	%o1, 64, %o1
	fsrc2	%f26, %f10
	bne,pt	%xcc, 1b
	 add	%o0, 0x40, %o0

	ba,a,pt	%xcc, 5b
END(__md5_process_block_crop)
