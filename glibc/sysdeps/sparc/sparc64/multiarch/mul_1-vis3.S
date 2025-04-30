! SPARC v9 64-bit VIS3 __mpn_mul_1 -- Multiply a limb vector with a single
! limb and store the product in a second limb vector.
!
! Copyright (C) 2013-2021 Free Software Foundation, Inc.
! This file is part of the GNU C Library.
! Contributed by David S. Miller <davem@davemloft.net>
!
! The GNU C Library is free software; you can redistribute it and/or
! modify it under the terms of the GNU Lesser General Public
! License as published by the Free Software Foundation; either
! version 2.1 of the License, or (at your option) any later version.
!
! The GNU C Library is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
! Lesser General Public License for more details.
!
! You should have received a copy of the GNU Lesser General Public
! License along with the GNU C Library; if not, see
! <https://www.gnu.org/licenses/>.

#include <sysdep.h>

#define res_ptr	%o0
#define s1_ptr	%o1
#define sz	%o2
#define s2_limb	%o3
#define carry	%o5
#define tmp1	%g1
#define tmp2	%g2
#define tmp3	%g3
#define tmp4	%o4

	.register	%g2,#scratch
	.register	%g3,#scratch
ENTRY(__mpn_mul_1_vis3)
	subcc	sz, 1, sz
	be	.Lfinal_limb
	 clr	carry

.Lloop:
	ldx	[s1_ptr + 0x00], tmp1
	ldx	[s1_ptr + 0x08], tmp4
	mulx	tmp1, s2_limb, tmp3
	add	s1_ptr, 0x10, s1_ptr
	umulxhi	tmp1, s2_limb, tmp2
	sub	sz, 2, sz
	mulx	tmp4, s2_limb, tmp1
	add	res_ptr, 0x10, res_ptr
	umulxhi	tmp4, s2_limb, tmp4
	addcc	carry, tmp3, tmp3
	stx	tmp3, [res_ptr - 0x10]
	addxc	%g0, tmp2, carry
	addcc	carry, tmp1, tmp1
	addxc	%g0, tmp4, carry
	brgz	sz, .Lloop
	 stx	tmp1, [res_ptr - 0x08]

	brlz,pt	sz, .Lfinish
	 nop

.Lfinal_limb:
	ldx	[s1_ptr + 0x00], tmp1
	mulx	tmp1, s2_limb, tmp3
	umulxhi	tmp1, s2_limb, tmp2
	addcc	carry, tmp3, tmp3
	addxc	%g0, tmp2, carry
	stx	tmp3, [res_ptr + 0x00]

.Lfinish:
	retl
	 mov	carry, %o0
END(__mpn_mul_1_vis3)
