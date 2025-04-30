/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

/* Please consult the file sysdeps/unix/sysv/linux/m68k/sysdep.h for
   more information about the value -4095 used below.*/

	.text
ENTRY (syscall)
	move.l 4(%sp), %d0	/* Load syscall number.  */
	_DOARGS_6 (28)		/* Frob arguments.  */
	trap &0			/* Do the system call.  */
	UNDOARGS_6		/* Unfrob arguments.  */
	cmp.l &-4095, %d0	/* Check %d0 for error.  */
	jcc SYSCALL_ERROR_LABEL	/* Jump to error handler if negative.  */
	rts			/* Return to caller.  */
PSEUDO_END (syscall)
