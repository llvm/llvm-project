/* Determine the length of a string.  For SPARC v9.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jan Vondrák <jvon4518@ss1000.ms.mff.cuni.cz>,
                  Jakub Jelinek <jj@ultra.linux.cz>, and
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

	.register	%g2, #scratch
	.register	%g3, #scratch

	.text
	.align		32
ENTRY(strlen)
	mov	%o0, %o1
	andn	%o0, 0x7, %o0

	ldx	[%o0], %o5
	and	%o1, 0x7, %g1
	mov	-1, %g5

	sethi	%hi(0x01010101), %o2
	sll	%g1, 3, %g1

	or	%o2, %lo(0x01010101), %o2
	srlx	%g5, %g1, %o3

	sllx	%o2, 32, %g1
	sethi	%hi(0x0000ff00), %g5

	orn	%o5, %o3, %o5
	or	%o2, %g1, %o2

	sllx	%o2, 7, %o3
10:	add	%o0, 8, %o0

	andn	%o3, %o5, %g1
	sub	%o5, %o2, %g2

	andcc	%g1, %g2, %g0
	be,a,pt	%xcc, 10b
	 ldx	[%o0], %o5
	srlx	%o5, 32, %g1

	andn	%o3, %g1, %o4
	sub	%g1, %o2, %g2

	add	%o0, 4, %g3
	andcc	%o4, %g2, %g0
	movne	%icc, %g1, %o5

	move	%icc, %g3, %o0
	or	%g5, %lo(0x0000ff00), %g5
	mov	3 - 8, %g2

	andcc	%o5, %g5, %g0
	srlx	%o5, 16, %g1
	move	%icc, 2 - 8, %g2

	andcc	%g1, 0xff, %g0
	srl	%o5, 24, %o5
	move	%icc, 1 - 8, %g2

	movrz	%o5, 0 - 8, %g2
	sub	%o0, %o1, %o0

	retl
	 add	%o0, %g2, %o0
END(strlen)
libc_hidden_builtin_def (strlen)
