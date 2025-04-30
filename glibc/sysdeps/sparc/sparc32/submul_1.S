! SPARC v8 __mpn_submul_1 -- Multiply a limb vector with a limb and
! subtract the result from a second limb vector.

! Copyright (C) 1992-2021 Free Software Foundation, Inc.

! This file is part of the GNU MP Library.

! The GNU MP Library is free software; you can redistribute it and/or modify
! it under the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation; either version 2.1 of the License, or (at your
! option) any later version.

! The GNU MP Library is distributed in the hope that it will be useful, but
! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
! or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
! License for more details.

! You should have received a copy of the GNU Lesser General Public License
! along with the GNU MP Library; see the file COPYING.LIB.  If not,
! see <https://www.gnu.org/licenses/>.


! INPUT PARAMETERS
! res_ptr	o0
! s1_ptr	o1
! size		o2
! s2_limb	o3

#include <sysdep.h>

ENTRY(__mpn_submul_1)
	sub	%g0,%o2,%o2		! negate ...
	sll	%o2,2,%o2		! ... and scale size
	sub	%o1,%o2,%o1		! o1 is offset s1_ptr
	sub	%o0,%o2,%g1		! g1 is offset res_ptr

	mov	0,%o0			! clear cy_limb

LOC(loop):
	ld	[%o1+%o2],%o4
	ld	[%g1+%o2],%g2
	umul	%o4,%o3,%o5
	rd	%y,%g3
	addcc	%o5,%o0,%o5
	addx	%g3,0,%o0
	subcc	%g2,%o5,%g2
	addx	%o0,0,%o0
	st	%g2,[%g1+%o2]

	addcc	%o2,4,%o2
	bne	LOC(loop)
	 nop

	retl
	 nop

END(__mpn_submul_1)
