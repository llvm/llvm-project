/* Copyright 2003-2021 Free Software Foundation, Inc.
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

#include <sys/asm.h>

/* Sign-extend the ioctl number, since the kernel wants it as a
   sign-extended 32-bit value, but our prototype is that of a long.  */

	.text
ENTRY (__ioctl)
	sll a1, a1, 0
	li v0, __NR_ioctl
	syscall			/* Do the system call.  */
	bne a3, zero, L(error)
	ret

L(error):
	SETUP_GP64_REG (a0, __ioctl)
	PTR_LA t9, __syscall_error
	RESTORE_GP64_REG
	jr t9

PSEUDO_END (__ioctl)

libc_hidden_def (__ioctl)
weak_alias (__ioctl, ioctl)
