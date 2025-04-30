/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

/* Please consult the file sysdeps/unix/sysv/linux/i386/sysdep.h for
   more information about the value -4095 used below.*/

	.text
ENTRY (syscall)

	PUSHARGS_6		/* Save register contents.  */
	_DOARGS_6(44)		/* Load arguments.  */
	movl 20(%esp), %eax	/* Load syscall number into %eax.  */
	ENTER_KERNEL		/* Do the system call.  */
	POPARGS_6		/* Restore register contents.  */
	cmpl $-4095, %eax	/* Check %eax for error.  */
	jae SYSCALL_ERROR_LABEL	/* Jump to error handler if error.  */
	ret			/* Return to caller.  */

PSEUDO_END (syscall)
