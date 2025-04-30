/* BSD `_setjmp' entry point to `sigsetjmp (..., 0)'.  MIPS version.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

/* This just does a tail-call to `__sigsetjmp (ARG, 0)'.
   We cannot do it in C because it must be a tail-call, so frame-unwinding
   in setjmp doesn't clobber the state restored by longjmp.  */

#include <sysdep.h>

	.set	nomips16

#ifdef __PIC__
	.option pic2
#endif
ENTRY (_setjmp)
#ifdef __PIC__
	.set	noreorder
	.cpload t9
	.set	reorder
	la	t9, C_SYMBOL_NAME (__sigsetjmp)
#endif
	move	a1,zero		/* Pass a second argument of zero.  */
#ifdef __PIC__
	jr	t9
#else
	j	C_SYMBOL_NAME (__sigsetjmp)
#endif
END (_setjmp)
libc_hidden_def (_setjmp)
