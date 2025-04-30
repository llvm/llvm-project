/* mpn_submul_1 -- multiply and subtract bignums.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

	.syntax unified
	.text

@		cycles/limb
@ StrongArm	   ?
@ Cortex-A8	   ?
@ Cortex-A9	   ?
@ Cortex-A15	   4

/* mp_limb_t mpn_submul_1(res_ptr, src1_ptr, size, s2_limb) */

ENTRY (__mpn_submul_1)
	push	{ r4, r5, r6, r7 }
	cfi_adjust_cfa_offset (16)
	cfi_rel_offset (r4, 0)
	cfi_rel_offset (r5, 4)
	cfi_rel_offset (r6, 8)
	cfi_rel_offset (r7, 12)

	ldr	r6, [r1], #4
	ldr	r7, [r0]
	mov	r4, #0			/* init carry in */
	b	1f
0:
	ldr	r6, [r1], #4		/* load next ul */
	adds	r5, r5, r4		/* (lpl, c) = lpl + cl */
	adc	r4, ip, #0		/* cl = hpl + c */
	subs	r5, r7, r5		/* (lpl, !c) = rl - lpl */
	ldr	r7, [r0, #4]		/* load next rl */
	it	cc
	addcc	r4, r4, #1		/* cl += !c */
	str	r5, [r0], #4
1:
	umull	r5, ip, r6, r3		/* (hpl, lpl) = ul * vl */
	subs	r2, r2, #1
	bne	0b

	adds	r5, r5, r4		/* (lpl, c) = lpl + cl */
	adc	r4, ip, #0		/* cl = hpl + c */
	subs	r5, r7, r5		/* (lpl, !c) = rl - lpl */
	str	r5, [r0], #4
	it	cc
	addcc	r4, r4, #1		/* cl += !c */
	mov	r0, r4			/* return carry */

	pop	{ r4, r5, r6, r7 }
	DO_RET	(lr)
END (__mpn_submul_1)
