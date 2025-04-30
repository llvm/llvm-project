/* Set a block of memory to some byte value.  For SUN4V Niagara-4.
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

#define ASI_BLK_INIT_QUAD_LDD_P	0xe2

#if IS_IN (libc)

	.register	%g2, #scratch
	.register	%g3, #scratch

	.text
	.align		32

ENTRY(__memset_niagara4)
	andcc		%o1, 0xff, %o4
	be,pt		%icc, 1f
	 mov		%o2, %o1
	sllx		%o4, 8, %g1
	or		%g1, %o4, %o2
	sllx		%o2, 16, %g1
	or		%g1, %o2, %o2
	sllx		%o2, 32, %g1
	ba,pt		%icc, 1f
	 or		%g1, %o2, %o4
END(__memset_niagara4)

	.align		32
ENTRY(__bzero_niagara4)
	clr		%o4
1:	cmp		%o1, 16
	ble		%icc, .Ltiny
	 mov		%o0, %o3
	sub		%g0, %o0, %g1
	and		%g1, 0x7, %g1
	brz,pt		%g1, .Laligned8
	 sub		%o1, %g1, %o1
1:	stb		%o4, [%o0 + 0x00]
	subcc		%g1, 1, %g1
	bne,pt		%icc, 1b
	 add		%o0, 1, %o0
.Laligned8:
	cmp		%o1, 64 + (64 - 8)
	ble		.Lmedium
	 sub		%g0, %o0, %g1
	andcc		%g1, (64 - 1), %g1
	brz,pn		%g1, .Laligned64
	 sub		%o1, %g1, %o1
1:	stx		%o4, [%o0 + 0x00]
	subcc		%g1, 8, %g1
	bne,pt		%icc, 1b
	 add		%o0, 0x8, %o0
.Laligned64:
	andn		%o1, 64 - 1, %g1
	sub		%o1, %g1, %o1
	brnz,pn		%o4, .Lnon_bzero_loop
	 mov		0x20, %g2
1:	stxa		%o4, [%o0 + %g0] ASI_BLK_INIT_QUAD_LDD_P
	subcc		%g1, 0x40, %g1
	stxa		%o4, [%o0 + %g2] ASI_BLK_INIT_QUAD_LDD_P
	bne,pt		%icc, 1b
	 add		%o0, 0x40, %o0
.Lpostloop:
	cmp		%o1, 8
	bl,pn		%icc, .Ltiny
	 membar		#StoreStore|#StoreLoad
.Lmedium:
	andn		%o1, 0x7, %g1
	sub		%o1, %g1, %o1
1:	stx		%o4, [%o0 + 0x00]
	subcc		%g1, 0x8, %g1
	bne,pt		%icc, 1b
	 add		%o0, 0x08, %o0
	andcc		%o1, 0x4, %g1
	be,pt		%icc, .Ltiny
	 sub		%o1, %g1, %o1
	stw		%o4, [%o0 + 0x00]
	add		%o0, 0x4, %o0
.Ltiny:
	cmp		%o1, 0
	be,pn		%icc, .Lexit
1:	 subcc		%o1, 1, %o1
	stb		%o4, [%o0 + 0x00]
	bne,pt		%icc, 1b
	 add		%o0, 1, %o0
.Lexit:
	retl
	 mov		%o3, %o0
.Lnon_bzero_loop:
	mov		0x08, %g3
	mov		0x28, %o5
1:	stxa		%o4, [%o0 + %g0] ASI_BLK_INIT_QUAD_LDD_P
	subcc		%g1, 0x40, %g1
	stxa		%o4, [%o0 + %g2] ASI_BLK_INIT_QUAD_LDD_P
	stxa		%o4, [%o0 + %g3] ASI_BLK_INIT_QUAD_LDD_P
	stxa		%o4, [%o0 + %o5] ASI_BLK_INIT_QUAD_LDD_P
	add		%o0, 0x10, %o0
	stxa		%o4, [%o0 + %g0] ASI_BLK_INIT_QUAD_LDD_P
	stxa		%o4, [%o0 + %g2] ASI_BLK_INIT_QUAD_LDD_P
	stxa		%o4, [%o0 + %g3] ASI_BLK_INIT_QUAD_LDD_P
	stxa		%o4, [%o0 + %o5] ASI_BLK_INIT_QUAD_LDD_P
	bne,pt		%icc, 1b
	 add		%o0, 0x30, %o0
	ba,a,pt		%icc, .Lpostloop
END(__bzero_niagara4)

#endif
