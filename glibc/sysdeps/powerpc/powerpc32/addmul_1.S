/* Multiply a limb vector by a single limb, for PowerPC.
   Copyright (C) 1993-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

/* mp_limb_t mpn_addmul_1 (mp_ptr res_ptr, mp_srcptr s1_ptr,
                           mp_size_t s1_size, mp_limb_t s2_limb)
   Calculate res+s1*s2 and put result back in res; return carry.  */
ENTRY (__mpn_addmul_1)
	mtctr	r5

	lwz	r0,0(r4)
	mullw	r7,r0,r6
	mulhwu	r10,r0,r6
	lwz     r9,0(r3)
	addc	r8,r7,r9
	addi	r3,r3,-4		/* adjust res_ptr */
	bdz	L(1)

L(0):	lwzu	r0,4(r4)
	stwu	r8,4(r3)
	mullw	r8,r0,r6
	adde	r7,r8,r10
	mulhwu	r10,r0,r6
	lwz     r9,4(r3)
	addze   r10,r10
	addc    r8,r7,r9
	bdnz	L(0)

L(1):	stw	r8,4(r3)
	addze	r3,r10
	blr
END (__mpn_addmul_1)
