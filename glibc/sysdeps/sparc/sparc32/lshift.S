! Sparc __mpn_lshift --
!
! Copyright (C) 1995-2021 Free Software Foundation, Inc.
!
! This file is part of the GNU MP Library.
!
! The GNU MP Library is free software; you can redistribute it and/or modify
! it under the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation; either version 2.1 of the License, or (at your
! option) any later version.
!
! The GNU MP Library is distributed in the hope that it will be useful, but
! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
! or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
! License for more details.
!
! You should have received a copy of the GNU Lesser General Public License
! along with the GNU MP Library; see the file COPYING.LIB.  If not,
! see <https://www.gnu.org/licenses/>.


! INPUT PARAMETERS
! RES_PTR	%o0
! SRC_PTR	%o1
! SIZE		%o2
! CNT		%o3

#include <sysdep.h>

ENTRY(__mpn_lshift)
	sll	%o2,2,%g1
	add	%o1,%g1,%o1	! make %o1 point at end of src
	ld	[%o1-4],%g2	! load first limb
	sub	%g0,%o3,%o5	! negate shift count
	add	%o0,%g1,%o0	! make %o0 point at end of res
	add	%o2,-1,%o2
	andcc	%o2,4-1,%g4	! number of limbs in first loop
	srl	%g2,%o5,%g1	! compute function result
	be	LOC(0)		! if multiple of 4 limbs, skip first loop
	st	%g1,[%sp+80]

	sub	%o2,%g4,%o2	! adjust count for main loop

LOC(loop0):
	ld	[%o1-8],%g3
	add	%o0,-4,%o0
	add	%o1,-4,%o1
	addcc	%g4,-1,%g4
	sll	%g2,%o3,%o4
	srl	%g3,%o5,%g1
	mov	%g3,%g2
	or	%o4,%g1,%o4
	bne	LOC(loop0)
	 st	%o4,[%o0+0]

LOC(0):	tst	%o2
	be	LOC(end)
	 nop

LOC(loop):
	ld	[%o1-8],%g3
	add	%o0,-16,%o0
	addcc	%o2,-4,%o2
	sll	%g2,%o3,%o4
	srl	%g3,%o5,%g1

	ld	[%o1-12],%g2
	sll	%g3,%o3,%g4
	or	%o4,%g1,%o4
	st	%o4,[%o0+12]
	srl	%g2,%o5,%g1

	ld	[%o1-16],%g3
	sll	%g2,%o3,%o4
	or	%g4,%g1,%g4
	st	%g4,[%o0+8]
	srl	%g3,%o5,%g1

	ld	[%o1-20],%g2
	sll	%g3,%o3,%g4
	or	%o4,%g1,%o4
	st	%o4,[%o0+4]
	srl	%g2,%o5,%g1

	add	%o1,-16,%o1
	or	%g4,%g1,%g4
	bne	LOC(loop)
	 st	%g4,[%o0+0]

LOC(end):
	sll	%g2,%o3,%g2
	st	%g2,[%o0-4]
	retl
	ld	[%sp+80],%o0

END(__mpn_lshift)
