/* syscall - indirect system call.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

/* We don't need a special syscall to implement syscall().  It won't work
   reliably with 64-bit arguments (but that is true on many modern platforms).
*/

ENTRY (syscall)
	mov	r2, r4
	mov	r4, r5
	mov	r5, r6
	mov	r6, r7
	ldw	r7, 0(sp)
	ldw	r8, 4(sp)
	ldw	r9, 8(sp)
	trap
	bne	r7, zero, SYSCALL_ERROR_LABEL
        ret
PSEUDO_END (syscall)
