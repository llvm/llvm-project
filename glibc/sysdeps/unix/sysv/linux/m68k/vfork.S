/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Schwab <schwab@gnu.org>.

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
#define _ERRNO_H	1
#include <bits/errno.h>
#include <tcb-offsets.h>

/* Clone the calling process, but without copying the whole address space.
   The calling process is suspended until the new process exits or is
   replaced by a call to `execve'.  Return -1 for errors, 0 to the new process,
   and the process ID of the new process to the old process.  */

ENTRY (__vfork)

	/* Pop the return PC value into A0.  */
	movel	%sp@+, %a0
	cfi_adjust_cfa_offset (-4)
	cfi_register (%pc, %a0)

	/* Stuff the syscall number in D0 and trap into the kernel.  */
	movel	#SYS_ify (vfork), %d0
	trap	#0

	tstl	%d0
	jmi	.Lerror		/* Branch forward if it failed.  */

	/* Jump to the return PC.  */
	jmp	%a0@

.Lerror:
	/* Push back the return PC.  */
	movel	%a0,%sp@-
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (%pc, 0)

#ifndef PIC
	jbra	SYSCALL_ERROR_LABEL
#endif

PSEUDO_END (__vfork)
libc_hidden_def (__vfork)

weak_alias (__vfork, vfork)
strong_alias (__vfork, __libc_vfork)
