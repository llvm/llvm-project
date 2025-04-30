! SPARC v8 __mpn_addmul_1 -- Multiply a limb vector with a limb and
! add the result to a second limb vector.

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

ENTRY(__mpn_addmul_1)
	ld	[%o1+0],%o4	! 1
	sll	%o2,4,%g1
	orcc	%g0,%g0,%g2
	mov	%o7,%g4			! Save return address register
	and	%g1,(4-1)<<4,%g1
1:	call	2f
	 add	%o7,3f-1b,%g3
2:	jmp	%g3+%g1
	 mov	%g4,%o7			! Restore return address register

	.align	4
3:
LOC(00):
	add	%o0,-4,%o0
	b	LOC(loop00)		/* 4, 8, 12, ... */
	 add	%o1,-4,%o1
	nop
LOC(01):
	b	LOC(loop01)		/* 1, 5, 9, ... */
	 nop
	nop
	nop
LOC(10):
	add	%o0,-12,%o0	/* 2, 6, 10, ... */
	b	LOC(loop10)
	 add	%o1,4,%o1
	nop
LOC(11):
	add	%o0,-8,%o0	/* 3, 7, 11, ... */
	b	LOC(loop11)
	 add	%o1,-8,%o1
	nop

LOC(loop):
	addcc	%g3,%g2,%g3	! 1
	ld	[%o1+4],%o4	! 2
	rd	%y,%g2		! 1
	addx	%g0,%g2,%g2
	ld	[%o0+0],%g1	! 2
	addcc	%g1,%g3,%g3
	st	%g3,[%o0+0]	! 1
LOC(loop00):
	umul	%o4,%o3,%g3	! 2
	ld	[%o0+4],%g1	! 2
	addxcc	%g3,%g2,%g3	! 2
	ld	[%o1+8],%o4	! 3
	rd	%y,%g2		! 2
	addx	%g0,%g2,%g2
	nop
	addcc	%g1,%g3,%g3
	st	%g3,[%o0+4]	! 2
LOC(loop11):
	umul	%o4,%o3,%g3	! 3
	addxcc	%g3,%g2,%g3	! 3
	ld	[%o1+12],%o4	! 4
	rd	%y,%g2		! 3
	add	%o1,16,%o1
	addx	%g0,%g2,%g2
	ld	[%o0+8],%g1	! 2
	addcc	%g1,%g3,%g3
	st	%g3,[%o0+8]	! 3
LOC(loop10):
	umul	%o4,%o3,%g3	! 4
	addxcc	%g3,%g2,%g3	! 4
	ld	[%o1+0],%o4	! 1
	rd	%y,%g2		! 4
	addx	%g0,%g2,%g2
	ld	[%o0+12],%g1	! 2
	addcc	%g1,%g3,%g3
	st	%g3,[%o0+12]	! 4
	add	%o0,16,%o0
	addx	%g0,%g2,%g2
LOC(loop01):
	addcc	%o2,-4,%o2
	bg	LOC(loop)
	 umul	%o4,%o3,%g3	! 1

	addcc	%g3,%g2,%g3	! 4
	rd	%y,%g2		! 4
	addx	%g0,%g2,%g2
	ld	[%o0+0],%g1	! 2
	addcc	%g1,%g3,%g3
	st	%g3,[%o0+0]	! 4
	retl
	 addx	%g0,%g2,%o0

END(__mpn_addmul_1)
