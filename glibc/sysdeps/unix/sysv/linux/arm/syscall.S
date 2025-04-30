/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

/* In the EABI syscall interface, we don't need a special syscall to
   implement syscall().  It won't work reliably with 64-bit arguments
   (but that is true on many modern platforms).  */

ENTRY (syscall)
	mov	ip, sp
	push	{r4, r5, r6, r7}
	cfi_adjust_cfa_offset (16)
	cfi_rel_offset (r4, 0)
	cfi_rel_offset (r5, 4)
	cfi_rel_offset (r6, 8)
	cfi_rel_offset (r7, 12)
	mov	r7, r0
	mov	r0, r1
	mov	r1, r2
	mov	r2, r3
	ldmfd	ip, {r3, r4, r5, r6}
	swi	0x0
	pop	{r4, r5, r6, r7}
	cfi_adjust_cfa_offset (-16)
	cfi_restore (r4)
	cfi_restore (r5)
	cfi_restore (r6)
	cfi_restore (r7)
	cmn	r0, #4096
	it	cc
	RETINSTR(cc, lr)
	b	PLTJMP(syscall_error)
PSEUDO_END (syscall)
