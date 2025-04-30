/* syscall - indirect system call.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

ENTRY (syscall)
	mov_s	r8, r0
	mov_s	r0, r1
	mov_s	r1, r2
	mov_s	r2, r3
	mov_s	r3, r4
	mov_s	r4, r5
	mov_s	r5, r6

	ARC_TRAP_INSN
	brhi	r0, -4096, L (call_syscall_err)
	j	[blink]
PSEUDO_END (syscall)
