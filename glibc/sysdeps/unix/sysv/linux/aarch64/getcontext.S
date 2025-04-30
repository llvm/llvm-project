/* Save current context.

   Copyright (C) 2009-2021 Free Software Foundation, Inc.

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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>
#include "ucontext_i.h"
#include "ucontext-internal.h"

/* int getcontext (ucontext_t *ucp)

   Returns 0 on success -1 and errno on failure.
 */

	.text

ENTRY(__getcontext)
	PTR_ARG (0)
	/* The saved context will return to the getcontext() call point
	   with a return value of 0 */
	str	xzr,	  [x0, oX0 +  0 * SZREG]

	stp	x18, x19, [x0, oX0 + 18 * SZREG]
	stp	x20, x21, [x0, oX0 + 20 * SZREG]
	stp	x22, x23, [x0, oX0 + 22 * SZREG]
	stp	x24, x25, [x0, oX0 + 24 * SZREG]
	stp	x26, x27, [x0, oX0 + 26 * SZREG]
	stp	x28, x29, [x0, oX0 + 28 * SZREG]
	str     x30,      [x0, oX0 + 30 * SZREG]

	/* Place LR into the saved PC, this will ensure that when
	   switching to this saved context with setcontext() control
	   will pass back to the caller of getcontext(), we have
	   already arrange to return the appropriate return value in x0
	   above.  */
	str	x30, [x0, oPC]

	/* Save the current SP */
	mov	x2, sp
	str     x2, [x0, oSP]

	/* Initialize the pstate.  */
	str	xzr, [x0, oPSTATE]

	/* Figure out where to place the first context extension
	   block.  */
	add     x2, x0, #oEXTENSION

	/* Write the context extension fpsimd header.  */
	mov	w3, #(FPSIMD_MAGIC & 0xffff)
	movk	w3, #(FPSIMD_MAGIC >> 16), lsl #16
	str	w3, [x2, #oHEAD + oMAGIC]
	mov	w3, #FPSIMD_CONTEXT_SIZE
	str	w3, [x2, #oHEAD + oSIZE]

	/* Fill in the FP SIMD context.  */
	add	x3, x2, #oV0 + 8 * SZVREG
	stp	 q8,  q9, [x3], # 2 * SZVREG
	stp	q10, q11, [x3], # 2 * SZVREG
	stp	q12, q13, [x3], # 2 * SZVREG
	stp	q14, q15, [x3], # 2 * SZVREG

	add	x3, x2, oFPSR

	mrs	x4, fpsr
	str	w4, [x3]

	mrs	x4, fpcr
	str	w4, [x3, oFPCR - oFPSR]

	/* Write the termination context extension header.  */
	add	x2, x2, #FPSIMD_CONTEXT_SIZE

	str	xzr, [x2, #oHEAD + oMAGIC]
	str	xzr, [x2, #oHEAD + oSIZE]

	/* Grab the signal mask */
	/* rt_sigprocmask (SIG_BLOCK, NULL, &ucp->uc_sigmask, _NSIG8) */
	add	PTR_REG (2), PTR_REG (0), #UCONTEXT_SIGMASK
	mov	x0, SIG_BLOCK
	mov	x1, 0
	mov	x3, _NSIG8
	mov	x8, SYS_ify (rt_sigprocmask)
	svc	0
	cbnz	x0, 1f

	/* Return 0 for success */
	mov	x0, 0
	RET
1:
	b	C_SYMBOL_NAME(__syscall_error)

	PSEUDO_END (__getcontext)
weak_alias (__getcontext, getcontext)
