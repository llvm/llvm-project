/* S390 __mpn_addmul_1 -- Multiply a limb vector with a limb and add
   the result to a second limb vector.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com).
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

/*
   INPUT PARAMETERS
   res_ptr	%r2
   s1_ptr	%r3
   sizeP	%r4
   s2_limb	%r5
*/

#include "sysdep.h"
#include "asm-syntax.h"

	.text
ENTRY(__mpn_addmul_1)
	st     %r6,24(%r15)
	cfi_offset (%r6, -72)
	slr    %r6,%r6            # cy_limb = 0
.L0:	icm    %r1,15,0(%r3)      # get s1_ptr[i]
	mr     %r0,%r5            # umul_ppmm(prod_high,prod_low,s1_ptr[j],s2_limb)
	jnm    .L1
	alr    %r0,%r5
.L1:	ltr    %r5,%r5
	jnm    .L2
	al     %r0,0(%r3)
.L2:	alr    %r1,%r6            # prod_low += cy_limb
	lr     %r6,%r0            # cy_limb = prod_high
	brc    12,.L3
	ahi    %r6,1              #           + (prod_low < cy_limb)
.L3:	al     %r1,0(%r2)         # prod_low += res_ptr[i]
	brc    12,.L4
	ahi    %r6,1              # cy_limb++
.L4:	st     %r1,0(%r2)
	la     %r2,4(0,%r2)
	la     %r3,4(0,%r3)
	brct   %r4,.L0
        lr     %r2,%r6            # return cy_limb
	l      %r6,24(%r15)
.Lexit:	br     %r14
END(__mpn_addmul_1)
