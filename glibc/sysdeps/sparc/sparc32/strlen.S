/* Determine the length of a string.
   For SPARC v7.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jj@ultra.linux.cz> and
                  David S. Miller <davem@davemloft.net>.

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
	.align		4

ENTRY(strlen)
	mov	%o0, %o1
	andn	%o0, 0x3, %o0

	ld	[%o0], %o5
	and	%o1, 0x3, %g1
	mov	-1, %g5

	sethi	%hi(0x01010101), %o2
	sll	%g1, 3, %g1

	or	%o2, %lo(0x01010101), %o2
	srl	%g5, %g1, %g2

	orn	%o5, %g2, %o5
	sll	%o2, 7, %o3
10:	add	%o0, 4, %o0

	andn	%o3, %o5, %g1
	sub	%o5, %o2, %g2

	andcc	%g1, %g2, %g0
	be,a	10b
	 ld	[%o0], %o5

	srl	%o5, 24, %g1

	andcc	%g1, 0xff, %g0
	be	90f
	 sub	%o0, 4, %o0

	srl	%o5, 16, %g2

	andcc	%g2, 0xff, %g0
	be	90f
	 add	%o0, 1, %o0

	srl	%o5, 8, %g1

	andcc	%g1, 0xff, %g0
	be	90f
	 add	%o0, 1, %o0

	add	%o0, 1, %o0

90:	retl
	 sub	%o0, %o1, %o0
END(strlen)
libc_hidden_builtin_def (strlen)
