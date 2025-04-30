! SPARC __mpn_add_n -- Add two limb vectors of the same length > 0 and store
! sum in a third limb vector.
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
#define RES_PTR	%o0
#define S1_PTR	%o1
#define S2_PTR	%o2
#define SIZE	%o3

#include <sysdep.h>

ENTRY(__mpn_add_n)
	xor	S2_PTR,RES_PTR,%g1
	andcc	%g1,4,%g0
	bne	LOC(1)			! branch if alignment differs
	 nop
! **  V1a  **
LOC(0):	andcc	RES_PTR,4,%g0		! RES_PTR unaligned? Side effect: cy=0
	be	LOC(v1)			! if no, branch
	 nop
/* Add least significant limb separately to align RES_PTR and S2_PTR */
	ld	[S1_PTR],%g4
	add	S1_PTR,4,S1_PTR
	ld	[S2_PTR],%g2
	add	S2_PTR,4,S2_PTR
	add	SIZE,-1,SIZE
	addcc	%g4,%g2,%o4
	st	%o4,[RES_PTR]
	add	RES_PTR,4,RES_PTR
LOC(v1):
	addx	%g0,%g0,%o4		! save cy in register
	cmp	SIZE,2			! if SIZE < 2 ...
	bl	LOC(end2)		! ... branch to tail code
	subcc	%g0,%o4,%g0		! restore cy

	ld	[S1_PTR+0],%g4
	addcc	SIZE,-10,SIZE
	ld	[S1_PTR+4],%g1
	ldd	[S2_PTR+0],%g2
	blt	LOC(fin1)
	subcc	%g0,%o4,%g0		! restore cy
/* Add blocks of 8 limbs until less than 8 limbs remain */
LOC(loop1):
	addxcc	%g4,%g2,%o4
	ld	[S1_PTR+8],%g4
	addxcc	%g1,%g3,%o5
	ld	[S1_PTR+12],%g1
	ldd	[S2_PTR+8],%g2
	std	%o4,[RES_PTR+0]
	addxcc	%g4,%g2,%o4
	ld	[S1_PTR+16],%g4
	addxcc	%g1,%g3,%o5
	ld	[S1_PTR+20],%g1
	ldd	[S2_PTR+16],%g2
	std	%o4,[RES_PTR+8]
	addxcc	%g4,%g2,%o4
	ld	[S1_PTR+24],%g4
	addxcc	%g1,%g3,%o5
	ld	[S1_PTR+28],%g1
	ldd	[S2_PTR+24],%g2
	std	%o4,[RES_PTR+16]
	addxcc	%g4,%g2,%o4
	ld	[S1_PTR+32],%g4
	addxcc	%g1,%g3,%o5
	ld	[S1_PTR+36],%g1
	ldd	[S2_PTR+32],%g2
	std	%o4,[RES_PTR+24]
	addx	%g0,%g0,%o4		! save cy in register
	addcc	SIZE,-8,SIZE
	add	S1_PTR,32,S1_PTR
	add	S2_PTR,32,S2_PTR
	add	RES_PTR,32,RES_PTR
	bge	LOC(loop1)
	subcc	%g0,%o4,%g0		! restore cy

LOC(fin1):
	addcc	SIZE,8-2,SIZE
	blt	LOC(end1)
	subcc	%g0,%o4,%g0		! restore cy
/* Add blocks of 2 limbs until less than 2 limbs remain */
LOC(loope1):
	addxcc	%g4,%g2,%o4
	ld	[S1_PTR+8],%g4
	addxcc	%g1,%g3,%o5
	ld	[S1_PTR+12],%g1
	ldd	[S2_PTR+8],%g2
	std	%o4,[RES_PTR+0]
	addx	%g0,%g0,%o4		! save cy in register
	addcc	SIZE,-2,SIZE
	add	S1_PTR,8,S1_PTR
	add	S2_PTR,8,S2_PTR
	add	RES_PTR,8,RES_PTR
	bge	LOC(loope1)
	subcc	%g0,%o4,%g0		! restore cy
LOC(end1):
	addxcc	%g4,%g2,%o4
	addxcc	%g1,%g3,%o5
	std	%o4,[RES_PTR+0]
	addx	%g0,%g0,%o4		! save cy in register

	andcc	SIZE,1,%g0
	be	LOC(ret1)
	subcc	%g0,%o4,%g0		! restore cy
/* Add last limb */
	ld	[S1_PTR+8],%g4
	ld	[S2_PTR+8],%g2
	addxcc	%g4,%g2,%o4
	st	%o4,[RES_PTR+8]

LOC(ret1):
	retl
	addx	%g0,%g0,%o0	! return carry-out from most sign. limb

LOC(1):	xor	S1_PTR,RES_PTR,%g1
	andcc	%g1,4,%g0
	bne	LOC(2)
	nop
! **  V1b  **
	mov	S2_PTR,%g1
	mov	S1_PTR,S2_PTR
	b	LOC(0)
	mov	%g1,S1_PTR

! **  V2  **
/* If we come here, the alignment of S1_PTR and RES_PTR as well as the
   alignment of S2_PTR and RES_PTR differ.  Since there are only two ways
   things can be aligned (that we care about) we now know that the alignment
   of S1_PTR and S2_PTR are the same.  */

LOC(2):	cmp	SIZE,1
	be	LOC(jone)
	nop
	andcc	S1_PTR,4,%g0		! S1_PTR unaligned? Side effect: cy=0
	be	LOC(v2)			! if no, branch
	nop
/* Add least significant limb separately to align S1_PTR and S2_PTR */
	ld	[S1_PTR],%g4
	add	S1_PTR,4,S1_PTR
	ld	[S2_PTR],%g2
	add	S2_PTR,4,S2_PTR
	add	SIZE,-1,SIZE
	addcc	%g4,%g2,%o4
	st	%o4,[RES_PTR]
	add	RES_PTR,4,RES_PTR

LOC(v2):
	addx	%g0,%g0,%o4		! save cy in register
	addcc	SIZE,-8,SIZE
	blt	LOC(fin2)
	subcc	%g0,%o4,%g0		! restore cy
/* Add blocks of 8 limbs until less than 8 limbs remain */
LOC(loop2):
	ldd	[S1_PTR+0],%g2
	ldd	[S2_PTR+0],%o4
	addxcc	%g2,%o4,%g2
	st	%g2,[RES_PTR+0]
	addxcc	%g3,%o5,%g3
	st	%g3,[RES_PTR+4]
	ldd	[S1_PTR+8],%g2
	ldd	[S2_PTR+8],%o4
	addxcc	%g2,%o4,%g2
	st	%g2,[RES_PTR+8]
	addxcc	%g3,%o5,%g3
	st	%g3,[RES_PTR+12]
	ldd	[S1_PTR+16],%g2
	ldd	[S2_PTR+16],%o4
	addxcc	%g2,%o4,%g2
	st	%g2,[RES_PTR+16]
	addxcc	%g3,%o5,%g3
	st	%g3,[RES_PTR+20]
	ldd	[S1_PTR+24],%g2
	ldd	[S2_PTR+24],%o4
	addxcc	%g2,%o4,%g2
	st	%g2,[RES_PTR+24]
	addxcc	%g3,%o5,%g3
	st	%g3,[RES_PTR+28]
	addx	%g0,%g0,%o4		! save cy in register
	addcc	SIZE,-8,SIZE
	add	S1_PTR,32,S1_PTR
	add	S2_PTR,32,S2_PTR
	add	RES_PTR,32,RES_PTR
	bge	LOC(loop2)
	subcc	%g0,%o4,%g0		! restore cy

LOC(fin2):
	addcc	SIZE,8-2,SIZE
	blt	LOC(end2)
	subcc	%g0,%o4,%g0		! restore cy
LOC(loope2):
	ldd	[S1_PTR+0],%g2
	ldd	[S2_PTR+0],%o4
	addxcc	%g2,%o4,%g2
	st	%g2,[RES_PTR+0]
	addxcc	%g3,%o5,%g3
	st	%g3,[RES_PTR+4]
	addx	%g0,%g0,%o4		! save cy in register
	addcc	SIZE,-2,SIZE
	add	S1_PTR,8,S1_PTR
	add	S2_PTR,8,S2_PTR
	add	RES_PTR,8,RES_PTR
	bge	LOC(loope2)
	subcc	%g0,%o4,%g0		! restore cy
LOC(end2):
	andcc	SIZE,1,%g0
	be	LOC(ret2)
	subcc	%g0,%o4,%g0		! restore cy
/* Add last limb */
LOC(jone):
	ld	[S1_PTR],%g4
	ld	[S2_PTR],%g2
	addxcc	%g4,%g2,%o4
	st	%o4,[RES_PTR]

LOC(ret2):
	retl
	addx	%g0,%g0,%o0	! return carry-out from most sign. limb

END(__mpn_add_n)
