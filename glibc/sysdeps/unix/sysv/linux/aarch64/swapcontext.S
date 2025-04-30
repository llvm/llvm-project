/* Modify saved context.

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

/* int swapcontext (ucontext_t *oucp, const ucontext_t *ucp) */

	.text
ENTRY(__swapcontext)
	PTR_ARG (0)
	/* Set the value returned when swapcontext() returns in this context.
	   And set up x1 to become the return address of the caller, so we
	   can return there with a normal RET instead of an indirect jump.  */
	stp	xzr, x30, [x0, oX0 +  0 * SZREG]
	/* Arrange the oucp context to return to 2f.  */
	adr	x30, 2f

	stp	x18, x19, [x0, oX0 + 18 * SZREG]
	stp	x20, x21, [x0, oX0 + 20 * SZREG]
	stp	x22, x23, [x0, oX0 + 22 * SZREG]
	stp	x24, x25, [x0, oX0 + 24 * SZREG]
	stp	x26, x27, [x0, oX0 + 26 * SZREG]
	stp	x28, x29, [x0, oX0 + 28 * SZREG]
	str     x30,      [x0, oX0 + 30 * SZREG]
	str     x30,      [x0, oPC]
	mov	x2, sp
	str     x2,       [x0, oSP]

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
	stp	 q8,  q9, [x3], #2 * SZVREG
	stp	q10, q11, [x3], #2 * SZVREG
	stp	q12, q13, [x3], #2 * SZVREG
	stp	q14, q15, [x3], #2 * SZVREG

	add	x3, x2, #oFPSR

	mrs	x4, fpsr
	str	w4, [x3, #oFPSR - oFPSR]

	mrs	x4, fpcr
	str	w4, [x3, #oFPCR - oFPSR]

	/* Write the termination context extension header.  */
	add	x2, x2, #FPSIMD_CONTEXT_SIZE

	str	xzr, [x2, #oHEAD + oMAGIC]
	str	xzr, [x2, #oHEAD + oSIZE]

	/* Preserve ucp.  */
	mov	x21, x1

	/* rt_sigprocmask (SIG_SETMASK, &ucp->uc_sigmask, &oucp->uc_sigmask,
			   _NSIG8) */
	/* Grab the signal mask */
	/* rt_sigprocmask (SIG_BLOCK, NULL, &ucp->uc_sigmask, _NSIG8) */
	add	x2, x0, #UCONTEXT_SIGMASK
	mov	x0, SIG_BLOCK
	mov	x1, 0
	mov	x3, _NSIG8
	mov	x8, SYS_ify (rt_sigprocmask)
	svc	0
	cbnz	x0, 1f

	mov     x22, x30
	mov	x0, x21
	bl	JUMPTARGET (__setcontext)
	mov     x30, x22
	RET

1:
	b	C_SYMBOL_NAME(__syscall_error)
2:
	/* The oucp context is restored here via an indirect branch,
	   x1 must be restored too which has the real return address.  */
	BTI_J
	mov	x30, x1
	RET
PSEUDO_END (__swapcontext)
weak_alias (__swapcontext, swapcontext)
