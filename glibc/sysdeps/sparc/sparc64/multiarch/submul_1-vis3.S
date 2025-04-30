! SPARC v9 64-bit VIS3 __mpn_submul_1 -- Multiply a limb vector with a
! limb and subtract the result from a second limb vector.
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

#define res_ptr	%i0
#define s1_ptr	%i1
#define sz	%i2
#define s2_limb	%i3
#define carry	%o5
#define tmp1	%g1
#define tmp2	%g2
#define tmp3	%g3
#define tmp4	%o4
#define tmp5	%l0
#define tmp6	%l1
#define tmp7	%l2
#define tmp8	%l3

	.register	%g2,#scratch
	.register	%g3,#scratch
ENTRY(__mpn_submul_1_vis3)
	save	%sp, -176, %sp
	subcc	sz, 1, sz
	be	.Lfinal_limb
	 clr	carry

.Lloop:
	ldx	[s1_ptr  + 0x00], tmp1
	ldx	[res_ptr + 0x00], tmp3
	ldx	[s1_ptr  + 0x08], tmp2
	ldx	[res_ptr + 0x08], tmp4
	mulx	tmp1, s2_limb, tmp5
	add	s1_ptr, 0x10, s1_ptr
	umulxhi	tmp1, s2_limb, tmp6
	add	res_ptr, 0x10, res_ptr
	mulx	tmp2, s2_limb, tmp7
	sub	sz, 2, sz
	umulxhi	tmp2, s2_limb, tmp8
	addcc	carry, tmp5, tmp5
	addxc	%g0, tmp6, carry
	subcc	tmp3, tmp5, tmp5
	addxc	%g0, carry, carry
	stx	tmp5, [res_ptr - 0x10]
	addcc	carry, tmp7, tmp7
	addxc	%g0, tmp8, carry
	subcc	tmp4, tmp7, tmp7
	addxc	%g0, carry, carry
	brgz	sz, .Lloop
	 stx	tmp7, [res_ptr - 0x08]

	brlz,pt	sz, .Lfinish
	 nop

.Lfinal_limb:
	ldx	[s1_ptr  + 0x00], tmp1
	ldx	[res_ptr + 0x00], tmp3
	mulx	tmp1, s2_limb, tmp5
	umulxhi	tmp1, s2_limb, tmp6
	addcc	carry, tmp5, tmp5
	addxc	%g0, tmp6, carry
	subcc	tmp3, tmp5, tmp5
	addxc	%g0, carry, carry
	stx	tmp5, [res_ptr + 0x00]

.Lfinish:
	jmpl	%i7 + 8, %g0
	 restore carry, 0, %o0
END(__mpn_submul_1_vis3)
