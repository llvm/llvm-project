/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

/* The following code is only used in the shared library when we
   compile the reentrant version.  Otherwise each system call defines
   each own version.  */

#ifndef PIC

/* The syscall stubs jump here when they detect an error.
   The code for Linux is almost identical to the canonical Unix
   code, except that the error number in %rax is negated.  */

#undef CALL_MCOUNT
#define CALL_MCOUNT /* Don't insert the profiling call, it clobbers %rax.  */

	.text
ENTRY (__syscall_error)
	neg %RAX_LP

#define __syscall_error __syscall_error_1
#include <sysdeps/unix/x86_64/sysdep.S>

#endif	/* !PIC */
