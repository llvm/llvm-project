/* BSD `_setjmp' entry point to `sigsetjmp (..., 0)'.  SH version.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

/* This just does a tail-call to `__sigsetjmp (ARG, 0)'.
   We cannot do it in C because it must be a tail-call, so frame-unwinding
   in setjmp doesn't clobber the state restored by longjmp.  */

#include <sysdep.h>

ENTRY (_setjmp)
#ifdef SHARED
	mova	1f, r0
	mov.l	1f, r1
	bra	2f
	 add	r1, r0
	.align	2
1:
	.long	_GLOBAL_OFFSET_TABLE_
2:
	mov.l	3f, r1
	mov.l	@(r0,r1), r1
	jmp	@r1
	 mov	#0, r5
	.align	2
3:
	.long	C_SYMBOL_NAME(__sigsetjmp@GOT)
#else
	mov.l	1f, r1
	jmp	@r1
	 mov	#0, r5
	.align	2
1:
	.long	C_SYMBOL_NAME(__sigsetjmp)
#endif
END (_setjmp)
libc_hidden_def (_setjmp)
