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


/* The following code is only used in the shared library when we
   compile the reentrant version.  Otherwise each system call defines
   each own version.  */

#ifndef	PIC

/* The syscall stubs jump here when they detect an error.  */

#undef CALL_MCOUNT
#define CALL_MCOUNT /* Don't insert the profiling call, it clobbers %d0.  */

	.text
ENTRY (__syscall_error)
	neg.l %d0
#ifndef _LIBC_REENTRANT
	move.l %d0, errno
#else
	move.l %d0, -(%sp)
	cfi_adjust_cfa_offset (4)
	jbsr __errno_location
	move.l (%sp)+, (%a0)
	cfi_adjust_cfa_offset (-4)
#endif
	move.l #-1, %d0
	/* Copy return value to %a0 for syscalls that are declared to
	   return a pointer.  */
	move.l %d0, %a0
	rts
END (__syscall_error)
#endif /* PIC */
