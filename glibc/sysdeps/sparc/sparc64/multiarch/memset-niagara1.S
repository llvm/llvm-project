/* Set a block of memory to some byte value.  For SUN4V Niagara.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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

#define ASI_BLK_INIT_QUAD_LDD_P	0xe2
#define ASI_P			0x80
#define ASI_PNF			0x82

#ifndef XCC
#define USE_BPR
#define XCC xcc
#endif

#if IS_IN (libc)

	.register	%g2,#scratch

	.text
	.align		32

ENTRY(__memset_niagara1)
	/* %o0=buf, %o1=pat, %o2=len */
	and		%o1, 0xff, %o3
	mov		%o2, %o1
	sllx		%o3, 8, %g1
	or		%g1, %o3, %o2
	sllx		%o2, 16, %g1
	or		%g1, %o2, %o2
	sllx		%o2, 32, %g1
	ba,pt		%XCC, 1f
	 or		%g1, %o2, %o2
END(__memset_niagara1)

ENTRY(__bzero_niagara1)
	clr		%o2
1:
# ifndef USE_BRP
	srl		%o1, 0, %o1
# endif
	brz,pn		%o1, 90f
	 mov		%o0, %o3

	wr		%g0, ASI_P, %asi

	cmp		%o1, 15
	blu,pn		%XCC, 70f
	 andcc		%o0, 0x7, %g1
	be,pt		%XCC, 2f
	 mov		8, %g2
	sub		%g2, %g1, %g1
	sub		%o1, %g1, %o1
1:	stba		%o2, [%o0 + 0x00] %asi
	subcc		%g1, 1, %g1
	bne,pt		%XCC, 1b
	 add		%o0, 1, %o0
2:	cmp		%o1, 128
	blu,pn		%XCC, 60f
	 andcc		%o0, (64 - 1), %g1
	be,pt		%XCC, 40f
	 mov		64, %g2
	sub		%g2, %g1, %g1
	sub		%o1, %g1, %o1
1:	stxa		%o2, [%o0 + 0x00] %asi
	subcc		%g1, 8, %g1
	bne,pt		%XCC, 1b
	 add		%o0, 8, %o0

40:
	wr		%g0, ASI_BLK_INIT_QUAD_LDD_P, %asi
	andn		%o1, (64 - 1), %g1
	sub		%o1, %g1, %o1

	andn		%g1, (256 - 1), %g2
	brz,pt		%g2, 50f
	 and		%g1, (256 - 1), %g1

45:
	stxa		%o2, [%o0 + 0x00] %asi
	stxa		%o2, [%o0 + 0x08] %asi
	stxa		%o2, [%o0 + 0x10] %asi
	stxa		%o2, [%o0 + 0x18] %asi
	stxa		%o2, [%o0 + 0x20] %asi
	stxa		%o2, [%o0 + 0x28] %asi
	stxa		%o2, [%o0 + 0x30] %asi
	stxa		%o2, [%o0 + 0x38] %asi
	stxa		%o2, [%o0 + 0x40] %asi
	stxa		%o2, [%o0 + 0x48] %asi
	stxa		%o2, [%o0 + 0x50] %asi
	stxa		%o2, [%o0 + 0x58] %asi
	stxa		%o2, [%o0 + 0x60] %asi
	stxa		%o2, [%o0 + 0x68] %asi
	stxa		%o2, [%o0 + 0x70] %asi
	stxa		%o2, [%o0 + 0x78] %asi
	stxa		%o2, [%o0 + 0x80] %asi
	stxa		%o2, [%o0 + 0x88] %asi
	stxa		%o2, [%o0 + 0x90] %asi
	stxa		%o2, [%o0 + 0x98] %asi
	stxa		%o2, [%o0 + 0xa0] %asi
	stxa		%o2, [%o0 + 0xa8] %asi
	stxa		%o2, [%o0 + 0xb0] %asi
	stxa		%o2, [%o0 + 0xb8] %asi
	stxa		%o2, [%o0 + 0xc0] %asi
	stxa		%o2, [%o0 + 0xc8] %asi
	stxa		%o2, [%o0 + 0xd0] %asi
	stxa		%o2, [%o0 + 0xd8] %asi
	stxa		%o2, [%o0 + 0xe0] %asi
	stxa		%o2, [%o0 + 0xe8] %asi
	stxa		%o2, [%o0 + 0xf0] %asi
	stxa		%o2, [%o0 + 0xf8] %asi
	subcc		%g2, 256, %g2
	bne,pt		%XCC, 45b
	 add		%o0, 256, %o0

	brz,pn		%g1, 55f
	 nop

50:
	stxa		%o2, [%o0 + 0x00] %asi
	stxa		%o2, [%o0 + 0x08] %asi
	stxa		%o2, [%o0 + 0x10] %asi
	stxa		%o2, [%o0 + 0x18] %asi
	stxa		%o2, [%o0 + 0x20] %asi
	stxa		%o2, [%o0 + 0x28] %asi
	stxa		%o2, [%o0 + 0x30] %asi
	stxa		%o2, [%o0 + 0x38] %asi
	subcc		%g1, 64, %g1
	bne,pt		%XCC, 50b
	 add		%o0, 64, %o0

55:
	wr		%g0, ASI_P, %asi
	brz,pn		%o1, 80f
60:
	 andncc		%o1, 0x7, %g1
	be,pn		%XCC, 2f
	 sub		%o1, %g1, %o1
1:	stxa		%o2, [%o0 + 0x00] %asi
	subcc		%g1, 8, %g1
	bne,pt		%XCC, 1b
	 add		%o0, 8, %o0
2:	brz,pt		%o1, 80f
	 nop

70:
1:	stba		%o2, [%o0 + 0x00] %asi
	subcc		%o1, 1, %o1
	bne,pt		%XCC, 1b
	 add		%o0, 1, %o0

	/* fallthrough */

80:
	wr		%g0, ASI_PNF, %asi

90:
	retl
	 mov		%o3, %o0
END(__bzero_niagara1)

#endif
