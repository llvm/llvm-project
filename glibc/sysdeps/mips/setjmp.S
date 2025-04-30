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

	.set	nomips16

/* The function __sigsetjmp_aux saves all the registers, but it can't
   reliably access the stack or frame pointers, so we pass them in as
   extra arguments.  */
#ifdef __PIC__
	.option pic2
#endif
ENTRY (__sigsetjmp)
#ifdef __PIC__
	.set	noreorder
	.cpload	t9
	.set	reorder
#endif
	move	a2, sp
#ifdef fp
	move	a3, fp
#else
	move	a3, $fp
#endif
#ifdef __PIC__
	la	t9, __sigsetjmp_aux
	jr	t9
#else
	j	__sigsetjmp_aux
#endif
END (__sigsetjmp)
hidden_def (__sigsetjmp)
