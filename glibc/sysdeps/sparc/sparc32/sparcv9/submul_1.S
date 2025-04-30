! SPARC v9 32-bit __mpn_submul_1 -- Multiply a limb vector with a limb
! and subtract the result from a second limb vector.
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
#define sz_arg	%i2
#define s2l_arg	%i3
#define sz	%o4
#define carry	%o5
#define s2_limb	%g1
#define tmp1	%l0
#define tmp2	%l1
#define tmp3	%l2
#define tmp4	%l3
#define tmp64_1	%g3
#define tmp64_2	%o3

ENTRY(__mpn_submul_1)
	save	%sp, -96, %sp
	srl	sz_arg, 0, sz
	srl	s2l_arg, 0, s2_limb
	subcc	sz, 1, sz
	be,pn	%icc, .Lfinal_limb
	 subcc	%g0, 0, carry

.Lloop:
	lduw	[s1_ptr  + 0x00], tmp1
	lduw	[res_ptr + 0x00], tmp3
	lduw	[s1_ptr  + 0x04], tmp2
	lduw	[res_ptr + 0x04], tmp4
	mulx	tmp1, s2_limb, tmp64_1
	add	s1_ptr, 8, s1_ptr
	mulx	tmp2, s2_limb, tmp64_2
	sub	sz, 2, sz
	add	res_ptr, 8, res_ptr
	addx	carry, tmp64_1, tmp64_1
	srlx	tmp64_1, 32, carry
	subcc	tmp3, tmp64_1, tmp64_1
	stw	tmp64_1, [res_ptr - 0x08]
	addx	carry, tmp64_2, tmp64_2
	srlx	tmp64_2, 32, carry
	subcc	tmp4, tmp64_2, tmp64_2
	brgz	sz, .Lloop
	 stw	tmp64_2, [res_ptr - 0x04]

	brlz,pt	sz, .Lfinish
	 nop

.Lfinal_limb:
	lduw	[s1_ptr + 0x00], tmp1
	lduw	[res_ptr + 0x00], tmp3
	mulx	tmp1, s2_limb, tmp64_1
	addx	carry, tmp64_1, tmp64_1
	srlx	tmp64_1, 32, carry
	subcc	tmp3, tmp64_1, tmp64_1
	stw	tmp64_1, [res_ptr + 0x00]

.Lfinish:
	addx	carry, 0, carry
	jmpl	%i7 + 0x8, %g0
	 restore carry, 0, %o0
END(__mpn_submul_1)
