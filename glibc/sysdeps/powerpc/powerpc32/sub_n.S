/* Subtract two limb vectors of equal, non-zero length for PowerPC.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

/* mp_limb_t mpn_sub_n (mp_ptr res_ptr, mp_srcptr s1_ptr, mp_srcptr s2_ptr,
                        mp_size_t size)
   Calculate s1-s2 and put result in res_ptr; return borrow, 0 or 1.  */

/* Note on optimisation: This code is optimal for the 601.  Almost every other
   possible 2-unrolled inner loop will not be.  Also, watch out for the
   alignment...  */

EALIGN (__mpn_sub_n, 3, 1)

/* Set up for loop below.  */
	mtcrf 0x01,r6
	srwi. r7,r6,1
	mtctr r7
	bt    31,L(2)

/* Set the carry (clear the borrow).  */
	subfc r0,r0,r0
/* Adjust pointers for loop.  */
	addi  r3,r3,-4
	addi  r4,r4,-4
	addi  r5,r5,-4
	b     L(0)

L(2):	lwz   r7,0(r5)
	lwz   r6,0(r4)
	subfc r6,r7,r6
	stw   r6,0(r3)
        beq   L(1)

/* Align start of loop to an odd word boundary to guarantee that the
   last two words can be fetched in one access (for 601).  This turns
   out to be important.  */
L(0):
	lwz   r9,4(r4)
	lwz   r8,4(r5)
	lwzu  r6,8(r4)
	lwzu  r7,8(r5)
	subfe r8,r8,r9
	stw   r8,4(r3)
	subfe r6,r7,r6
	stwu  r6,8(r3)
	bdnz  L(0)
/* Return the borrow. */
L(1):	subfe r3,r3,r3
	neg   r3,r3
	blr
END (__mpn_sub_n)
