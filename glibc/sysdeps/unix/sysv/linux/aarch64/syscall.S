/* Copyright (C) 2005-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

/* syscall (int nr, ...)

   AArch64 system calls take between 0 and 7 arguments. On entry here nr
   is in w0 and any other system call arguments are in register x1..x7.

   For kernel entry we need to move the system call nr to x8 then
   load the remaining arguments to register. */

ENTRY (syscall)
	uxtw	x8, w0
	mov	x0, x1
	mov	x1, x2
	mov	x2, x3
	mov	x3, x4
	mov	x4, x5
	mov	x5, x6
	mov	x6, x7
	svc	0x0
	cmn	x0, #4095
	b.cs	1f
	RET
1:
	b	SYSCALL_ERROR
PSEUDO_END (syscall)
