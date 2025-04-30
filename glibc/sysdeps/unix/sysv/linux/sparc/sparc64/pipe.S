/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Miguel de Icaza <miguel@gnu.ai.mit.edu>, 1997.

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

	.text

	.globl	__syscall_error
ENTRY(__libc_pipe)
	mov	%o0, %o2		/* Save PIPEDES. */
	LOADSYSCALL(pipe)
	ta	0x6d
	bcc,pt	%xcc, 1f
	 mov	%o7, %g1
	call	__syscall_error
	 mov	%g1, %o7
1:	st	%o0, [%o2]		/* PIPEDES[0] = %o0; */
	st	%o1, [%o2 + 4]		/* PIPEDES[1] = %o1; */
	retl
	 clr	%o0
END(__libc_pipe)

weak_alias (__libc_pipe, __pipe)
libc_hidden_def (__pipe)
weak_alias (__libc_pipe, pipe)
