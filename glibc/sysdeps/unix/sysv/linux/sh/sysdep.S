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

/* The syscall stubs jump here when they detect an error.
   The code for Linux is almost identical to the canonical Unix
   code, except that the error number in R0 is negated.  */

#undef CALL_MCOUNT
#define CALL_MCOUNT /* Don't insert the profiling call, it clobbers R0.  */

ENTRY (__syscall_error)
	neg r4, r0
	cfi_endproc

#define __syscall_error __syscall_error_1
#include <sysdeps/unix/sh/sysdep.S>
