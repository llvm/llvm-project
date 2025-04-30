! SPARC v9 32-bit __mpn_mul_1 -- Multiply a limb vector with a single
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

ENTRY(__mpn_mul_1)
	srl	sz, 0, sz
	srl	s2_limb, 0, s2_limb
	subcc	sz, 1, sz
	be,pn	%icc, .Lfinal_limb
	 clr	carry

.Lloop:
	lduw	[s1_ptr + 0x00], tmp1
	lduw	[s1_ptr + 0x04], tmp2
	mulx	tmp1, s2_limb, tmp3
	add	s1_ptr, 8, s1_ptr
	mulx	tmp2, s2_limb, tmp4
	sub	sz, 2, sz
	add	res_ptr, 8, res_ptr
	add	carry, tmp3, tmp3
	stw	tmp3, [res_ptr - 0x08]
	srlx	tmp3, 32, carry
	add	carry, tmp4, tmp4
	stw	tmp4, [res_ptr - 0x04]
	brgz	sz, .Lloop
	 srlx	tmp4, 32, carry

	brlz,pt	sz, .Lfinish
	 nop

.Lfinal_limb:
	lduw	[s1_ptr + 0x00], tmp1
	mulx	tmp1, s2_limb, tmp3
	add	carry, tmp3, tmp3
	stw	tmp3, [res_ptr + 0x00]
	srlx	tmp3, 32, carry

.Lfinish:
	retl
	 mov	carry, %o0
END(__mpn_mul_1)
