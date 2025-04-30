/* Record the register state before and after a variant PCS call.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

	.variant_pcs	vpcs_call
	.global	vpcs_call
	.type	vpcs_call, %function
vpcs_call:
	.cfi_startproc
	hint	34 /* bti c.  */

	/* Save register state to *x0.  */
	stp	x0, x1, [x0]
	stp	x2, x3, [x0, 16]
	stp	x4, x5, [x0, 32]
	stp	x6, x7, [x0, 48]
	stp	x8, x9, [x0, 64]
	stp	x10, x11, [x0, 80]
	stp	x12, x13, [x0, 96]
	stp	x14, x15, [x0, 112]
	stp	x16, x17, [x0, 128]
	stp	x18, x19, [x0, 144]
	stp	x20, x21, [x0, 160]
	stp	x22, x23, [x0, 176]
	stp	x24, x25, [x0, 192]
	stp	x26, x27, [x0, 208]
	stp	x28, x29, [x0, 224]
	mov	x1, sp
	stp	x30, x1, [x0, 240]
	stp	q0, q1, [x0, 256]
	stp	q2, q3, [x0, 288]
	stp	q4, q5, [x0, 320]
	stp	q6, q7, [x0, 352]
	stp	q8, q9, [x0, 384]
	stp	q10, q11, [x0, 416]
	stp	q12, q13, [x0, 448]
	stp	q14, q15, [x0, 480]
	stp	q16, q17, [x0, 512]
	stp	q18, q19, [x0, 544]
	stp	q20, q21, [x0, 576]
	stp	q22, q23, [x0, 608]
	stp	q24, q25, [x0, 640]
	stp	q26, q27, [x0, 672]
	stp	q28, q29, [x0, 704]
	stp	q30, q31, [x0, 736]
	ret
	.cfi_endproc
	.size	vpcs_call, .-vpcs_call

	.global	vpcs_call_regs
	.type	vpcs_call_regs, %function
vpcs_call_regs:
	.cfi_startproc
	hint	34 /* bti c.  */

	stp     x29, x30, [sp, -160]!
	mov     x29, sp

	/* Save callee-saved registers.  */
	stp	x19, x20, [sp, 16]
	stp	x21, x22, [sp, 32]
	stp	x23, x24, [sp, 48]
	stp	x25, x26, [sp, 64]
	stp	x27, x28, [sp, 80]
	stp	d8, d9, [sp, 96]
	stp	d10, d11, [sp, 112]
	stp	d12, d13, [sp, 128]
	stp	d14, d15, [sp, 144]

	/* Initialize most registers from *x1, and save x0, x1, x29, x30,
	   and sp (== x29), so *x1 contains the register state.  */
	stp	x0, x1, [x1]
	str	x29, [x1, 232]
	ldp	x2, x3, [x1, 16]
	ldp	x4, x5, [x1, 32]
	ldp	x6, x7, [x1, 48]
	ldp	x8, x9, [x1, 64]
	ldp	x10, x11, [x1, 80]
	ldp	x12, x13, [x1, 96]
	ldp	x14, x15, [x1, 112]
	ldp	x16, x17, [x1, 128]
	ldp	x18, x19, [x1, 144]
	ldp	x20, x21, [x1, 160]
	ldp	x22, x23, [x1, 176]
	ldp	x24, x25, [x1, 192]
	ldp	x26, x27, [x1, 208]
	ldr	x28, [x1, 224]
	/* Skip x29, x30, sp.  */
	ldp	q0, q1, [x1, 256]
	ldp	q2, q3, [x1, 288]
	ldp	q4, q5, [x1, 320]
	ldp	q6, q7, [x1, 352]
	ldp	q8, q9, [x1, 384]
	ldp	q10, q11, [x1, 416]
	ldp	q12, q13, [x1, 448]
	ldp	q14, q15, [x1, 480]
	ldp	q16, q17, [x1, 512]
	ldp	q18, q19, [x1, 544]
	ldp	q20, q21, [x1, 576]
	ldp	q22, q23, [x1, 608]
	ldp	q24, q25, [x1, 640]
	ldp	q26, q27, [x1, 672]
	ldp	q28, q29, [x1, 704]
	ldp	q30, q31, [x1, 736]

	/* Emulate a BL using B, but save x30 before the branch.  */
	adr	x30, .L_return_addr
	stp	x30, x29, [x1, 240]
	b	vpcs_call
.L_return_addr:

	/* Restore callee-saved registers.  */
	ldp	x19, x20, [sp, 16]
	ldp	x21, x22, [sp, 32]
	ldp	x23, x24, [sp, 48]
	ldp	x25, x26, [sp, 64]
	ldp	x27, x28, [sp, 80]
	ldp	d8, d9, [sp, 96]
	ldp	d10, d11, [sp, 112]
	ldp	d12, d13, [sp, 128]
	ldp	d14, d15, [sp, 144]

	ldp     x29, x30, [sp], 160
	ret
	.cfi_endproc
	.size	vpcs_call_regs, .-vpcs_call_regs
