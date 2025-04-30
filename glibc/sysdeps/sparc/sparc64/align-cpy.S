/* Aligned copy routines specified by Sparc V9 ABI.
   For 64-bit sparc.
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
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

	.text
	.align	8
ENTRY(__align_cpy_8)
10:	cmp	%o0, %o1
	be,pn	%xcc, 9f
	 mov	%o0, %o3
	subcc	%o2, 0x08, %o2
	be,pn	%xcc, 8f
1:	 ldx	[%o1 + 0x00], %o5
	ldx	[%o1 + 0x08], %o4
	subcc	%o2, 0x10, %o2
	add	%o1, 0x10, %o1
	stx	%o5, [%o3 + 0x00]
	stx	%o4, [%o3 + 0x08]
	bg,pt	%xcc, 1b
	 add	%o3, 0x10, %o3
	bne,pn	%xcc, 9f
	 nop
	ldx	[%o1 + 0x00], %o5
8:	stx	%o5, [%o3 + 0x00]
9:	retl
	 nop
END(__align_cpy_8)

	.align	8
ENTRY(__align_cpy_4)
20:	cmp	%o0, %o1
	be,pn	%xcc, 9f
	 mov	%o0, %o3
	subcc	%o2, 0x04, %o2
	be,pn	%xcc, 8f
1:	 lduw	[%o1 + 0x00], %o5
	lduw	[%o1 + 0x04], %o4
	subcc	%o2, 0x08, %o2
	add	%o1, 0x08, %o1
	stw	%o5, [%o3 + 0x00]
	stw	%o4, [%o3 + 0x04]
	bg,pt	%xcc, 1b
	 add	%o3, 0x08, %o3
	bne,pn	%xcc, 9f
	 nop
	lduw	[%o1 + 0x00], %o5
8:	stw	%o5, [%o3 + 0x00]
9:	retl
	 nop
END(__align_cpy_4)

	.align	8
ENTRY(__align_cpy_2)
	or	%o0, %o1, %o3
	or	%o2, %o3, %o3
	andcc	%o3, 0x7, %g0
	be,pt	%xcc, 10b
	 andcc	%o3, 0x3, %g0
	be,pt	%xcc, 20b
	 mov	%o7, %g1
	call	HIDDEN_JUMPTARGET(memcpy)
	 mov	%o7, %g1
END(__align_cpy_2)

weak_alias (__align_cpy_8, __align_cpy_16)
weak_alias (__align_cpy_2, __align_cpy_1)
