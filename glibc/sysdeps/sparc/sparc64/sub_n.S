/* SPARC v9 __mpn_sub_n -- Subtract two limb vectors of the same length > 0
   and store difference in a third limb vector.

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
   res_ptr	%o0
   s1_ptr	%o1
   s2_ptr	%o2
   size		%o3  */

ENTRY(__mpn_sub_n)

	sub %g0,%o3,%g5
	sllx %o3,3,%g1
	add %o1,%g1,%o1			! make s1_ptr point at end
	add %o2,%g1,%o2			! make s2_ptr point at end
	add %o0,%g1,%o0			! make res_ptr point at end
	mov 0,%o4			! clear carry variable
	sllx %g5,3,%o5			! compute initial address index

1:	ldx [%o2+%o5],%g1		! load s2 limb
	add %g5,1,%g5			! increment loop count
	ldx [%o1+%o5],%o3		! load s1 limb
	addcc %g1,%o4,%g1		! add s2 limb and carry variable
	movcc %xcc,0,%o4		! if carry-out, o4 was 1; clear it
	subcc %o3,%g1,%g1		! subtract s1 limb from sum
	stx %g1,[%o0+%o5]		! store result
	add %o5,8,%o5			! increment address index
	brnz,pt %g5,1b
	 movcs %xcc,1,%o4		! if s1 subtract gave carry, record it

	retl
	 mov %o4,%o0

END(__mpn_sub_n)
