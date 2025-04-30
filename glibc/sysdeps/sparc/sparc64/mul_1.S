/* SPARC v9 __mpn_mul_1 -- Multiply a limb vector with a single limb and
   store the product in a second limb vector.

   Copyright (C) 1995-2021 Free Software Foundation, Inc.

   This file is part of the GNU MP Library.

   The GNU MP Library is free software; you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation; either version 2.1 of the License, or (at your
   option) any later version.

   The GNU MP Library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
   License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with the GNU MP Library; see the file COPYING.LIB.  If not,
   see <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>


/* INPUT PARAMETERS
   res_ptr	o0
   s1_ptr	o1
   size		o2
   s2_limb	o3  */

ENTRY(__mpn_mul_1)
	!#PROLOGUE#	0
	save	%sp,-192,%sp
	!#PROLOGUE#	1

	sub	%g0,%i2,%o7
	sllx	%o7,3,%g5
	sub	%i1,%g5,%o3
	sub	%i0,%g5,%o4
	mov	0,%o0			! zero cy_limb

	srl	%i3,0,%o1		! extract low 32 bits of s2_limb
	srlx	%i3,32,%i3		! extract high 32 bits of s2_limb
	mov	1,%o2
	sllx	%o2,32,%o2		! o2 = 0x100000000

	!   hi   !
             !  mid-1 !
             !  mid-2 !
		 !   lo   !
.Loop:
	sllx	%o7,3,%g1
	ldx	[%o3+%g1],%g5
	srl	%g5,0,%i0		! zero hi bits
	srlx	%g5,32,%g5
	mulx	%o1,%i0,%i4		! lo product
	mulx	%i3,%i0,%i1		! mid-1 product
	mulx	%o1,%g5,%l2		! mid-2 product
	mulx	%i3,%g5,%i5		! hi product
	srlx	%i4,32,%i0		! extract high 32 bits of lo product...
	add	%i1,%i0,%i1		! ...and add it to the mid-1 product
	addcc	%i1,%l2,%i1		! add mid products
	mov	0,%l0			! we need the carry from that add...
	movcs	%xcc,%o2,%l0		! ...compute it and...
	add	%i5,%l0,%i5		! ...add to bit 32 of the hi product
	sllx	%i1,32,%i0		! align low bits of mid product
	srl	%i4,0,%g5		! zero high 32 bits of lo product
	add	%i0,%g5,%i0		! combine into low 64 bits of result
	srlx	%i1,32,%i1		! extract high bits of mid product...
	add	%i5,%i1,%i1		! ...and add them to the high result
	addcc	%i0,%o0,%i0		! add cy_limb to low 64 bits of result
	mov	0,%g5
	movcs	%xcc,1,%g5
	addcc	%o7,1,%o7
	stx	%i0,[%o4+%g1]
	bne,pt	%xcc,.Loop
	 add	%i1,%g5,%o0		! compute new cy_limb

	jmpl	%i7+8,%g0
	 restore %o0,%g0,%o0

END(__mpn_mul_1)
