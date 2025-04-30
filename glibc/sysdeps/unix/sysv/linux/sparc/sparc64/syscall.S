/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
	.text

	.globl	__syscall_error
ENTRY(syscall)
	mov	%o0,%g1
	mov	%o1,%o0
	mov	%o2,%o1
	mov	%o3,%o2
	mov	%o4,%o3
	mov	%o5,%o4
	ldx	[%sp + STACK_BIAS + 176],%o5

	ta	0x6d

	bcc,pt	%xcc, 1f
	 mov	%o7, %g1
	call	__syscall_error
	 mov	%g1, %o7
1:	retl
	 nop
END(syscall)
