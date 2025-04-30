/* mpn_add_n -- add (or subtract) bignums.
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
#include <arm-features.h>

	.syntax unified
	.text

#ifdef USE_AS_SUB_N
# define INITC	cmp r0, r0
# define OPC	sbcs
# define RETC	sbc r0, r0, r0; neg r0, r0
# define FUNC	__mpn_sub_n
#else
# define INITC	cmn r0, #0
# define OPC	adcs
# define RETC	mov r0, #0; adc r0, r0, r0
# define FUNC	__mpn_add_n
#endif

/* mp_limb_t mpn_add_n(res_ptr, src1_ptr, src2_ptr, size) */

ENTRY (FUNC)
	push	{ r4, r5, r6, r7, r8, r10, lr }
	cfi_adjust_cfa_offset (28)
	cfi_rel_offset (r4, 0)
	cfi_rel_offset (r5, 4)
	cfi_rel_offset (r6, 8)
	cfi_rel_offset (r7, 12)
	cfi_rel_offset (r8, 16)
	cfi_rel_offset (r10, 20)
	cfi_rel_offset (lr, 24)

	INITC				/* initialize carry flag */
	tst	r3, #1			/* count & 1 == 1? */
	add	lr, r1, r3, lsl #2	/* compute end src1 */
	beq	1f

	ldr	r4, [r1], #4		/* do one to make count even */
	ldr	r5, [r2], #4
	OPC	r4, r4, r5
	teq	r1, lr			/* end of count? (preserve carry) */
	str	r4, [r0], #4
	beq	9f
1:
	tst	r3, #2			/* count & 2 == 2?  */
	beq	2f
	ldm	r1!, { r4, r5 }		/* do two to make count 0 mod 4 */
	ldm	r2!, { r6, r7 }
	OPC	r4, r4, r6
	OPC	r5, r5, r7
	teq	r1, lr			/* end of count? */
	stm	r0!, { r4, r5 }
	beq	9f
2:
	ldm	r1!, { r3, r5, r7, r10 }	/* do four each loop */
	ldm	r2!, { r4, r6, r8, ip }
	OPC	r3, r3, r4
	OPC	r5, r5, r6
	OPC	r7, r7, r8
	OPC	r10, r10, ip
	teq	r1, lr
	stm	r0!, { r3, r5, r7, r10 }
	bne	2b

9:
	RETC				/* copy carry out */
#ifndef ARM_ALWAYS_BX
	pop	{ r4, r5, r6, r7, r8, r10, pc }
#else
	pop	{ r4, r5, r6, r7, r8, r10, lr }
	bx	lr
#endif
END (FUNC)
