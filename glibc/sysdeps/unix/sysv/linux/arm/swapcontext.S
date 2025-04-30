/* Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include "ucontext_i.h"

	.syntax unified
	.text

/* int swapcontext (ucontext_t *oucp, const ucontext_t *ucp) */

ENTRY(swapcontext)

	/* Have getcontext() do most of the work then fix up
	   LR afterwards.  Save R3 to keep the stack aligned.  */
	push	{r0,r1,r3,r14}
	cfi_adjust_cfa_offset (16)
	cfi_rel_offset (r0,0)
	cfi_rel_offset (r1,4)
	cfi_rel_offset (r3,8)
	cfi_rel_offset (r14,12)

	bl	__getcontext
	mov	r4, r0

	pop	{r0,r1,r3,r14}
	cfi_adjust_cfa_offset (-16)
	cfi_restore (r0)
	cfi_restore (r1)
	cfi_restore (r3)
	cfi_restore (r14)

	/* Exit if getcontext() failed.  */
	cmp 	r4, #0
	itt	ne
	movne	r0, r4
	RETINSTR(ne, r14)

	/* Fix up LR and the PC.  */
	str	r13,[r0, #MCONTEXT_ARM_SP]
	str	r14,[r0, #MCONTEXT_ARM_LR]
	str	r14,[r0, #MCONTEXT_ARM_PC]

	/* And swap using swapcontext().  */
	mov	r0, r1
	b	__setcontext

END(swapcontext)
