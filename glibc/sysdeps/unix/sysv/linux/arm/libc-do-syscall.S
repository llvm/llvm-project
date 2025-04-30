/* Copyright (C) 2010-2021 Free Software Foundation, Inc.
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

/* When this file is built for inclusion in nscd, _FORTIFY_SOURCE is
   defined but -O is not used, resulting in a warning from features.h.  */
#undef _FORTIFY_SOURCE
#include <sysdep.h>

/* Out-of-line syscall stub.  We expect the system call number in ip
   and return the raw result in r0.  No registers are clobbered.
   We could avoid using the stack for this, but the goal is accurate
   unwind information - and while there is a reserved prefix in the
   ARM unwind tables for register to register moves, the actual opcodes
   are not defined.  */

#if defined(__thumb__)
	.thumb
	.syntax unified
	.hidden __libc_do_syscall

#undef CALL_MCOUNT
#define CALL_MCOUNT

ENTRY (__libc_do_syscall)
	.fnstart
	push	{r7, lr}
	.save	{r7, lr}
	cfi_adjust_cfa_offset (8)
	cfi_rel_offset (r7, 0)
	cfi_rel_offset (lr, 4)
	mov	r7, ip
	swi	0x0
	pop	{r7, pc}
	.fnend
END (__libc_do_syscall)

#endif /* __thumb__ */
