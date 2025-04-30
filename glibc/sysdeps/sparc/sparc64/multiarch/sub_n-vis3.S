! SPARC v9 64-bit VIS3 __mpn_sub_n -- Subtract two limb vectors of the same length > 0
! and store difference in a third limb vector.
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
#define s2_ptr	%o2
#define sz	%o3
#define tmp1	%g1
#define tmp2	%g2
#define tmp3	%g3
#define tmp4	%o4

	.register	%g2,#scratch
	.register	%g3,#scratch
ENTRY(__mpn_sub_n_vis3)
	subcc	sz, 1, sz
	be	.Lfinal_limb
	 cmp	%g0, 1

.Lloop:
	ldx	[s2_ptr + 0x00], tmp1
	add	s2_ptr, 0x10, s2_ptr
	ldx	[s1_ptr + 0x00], tmp2
	add	s1_ptr, 0x10, s1_ptr
	ldx	[s2_ptr - 0x08], tmp3
	add	res_ptr, 0x10, res_ptr
	ldx	[s1_ptr - 0x08], tmp4
	sub	sz, 2, sz
	xnor	tmp1, %g0, tmp1
	addxccc	tmp1, tmp2, tmp1
	stx	tmp1, [res_ptr - 0x10]
	xnor	tmp3, %g0, tmp3
	addxccc	tmp3, tmp4, tmp3
	brgz	sz, .Lloop
	stx	tmp3, [res_ptr - 0x08]

	brlz,pt	sz, .Lfinish
	 nop

.Lfinal_limb:
	ldx	[s2_ptr + 0x00], tmp1
	ldx	[s1_ptr + 0x00], tmp2
	xnor	tmp1, %g0, tmp1
	addxccc	tmp1, tmp2, tmp1
	stx	tmp1, [res_ptr + 0x00]

.Lfinish:
	clr	%o0
	retl
	 movcc	%xcc, 1, %o0
END(__mpn_sub_n_vis3)
