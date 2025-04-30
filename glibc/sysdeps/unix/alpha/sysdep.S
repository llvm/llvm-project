/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Brendan Kehoe (brendan@zen.org).

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
#include <features.h>

#if defined(PIC)
	/* Put this at the end of libc's text segment so that all of
	   the direct branches from the syscalls are forward, and
	   thus predicted not taken.  */
	.section .text.last, "ax", @progbits
#else
	.text
#endif

#if IS_IN (libc)
# define SYSCALL_ERROR_ERRNO __libc_errno
#else
# define SYSCALL_ERROR_ERRNO errno
#endif

	.align 4
	.globl	__syscall_error
	.ent	__syscall_error
__syscall_error:
	/* When building a shared library, we branch here without having
	   loaded the GP.  Nor, since it was a direct branch, have we
	   loaded PV with our address.

	   When building a static library, we tail call here from another
	   object file, possibly with a different GP, and must return with
	   the GP of our caller in place so that linker relaxation works.

	   Both issues are solved by computing the GP into T1 instead of
	   clobbering the traditional GP register.  */
	.prologue 0
	mov	v0, t0
	br	t1, 1f
1:	ldah	t1, 0(t1) !gpdisp!1
	call_pal PAL_rduniq

	lda	t1, 0(t1) !gpdisp!1
	ldq	t1, SYSCALL_ERROR_ERRNO(t1) !gottprel
	addq	v0, t1, t1
	lda	v0, -1

	stl	t0, 0(t1)
	ret

	.end __syscall_error
