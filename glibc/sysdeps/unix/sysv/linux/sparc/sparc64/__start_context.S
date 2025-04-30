/* Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

/* This is the helper code which gets called if a function which is
   registered with 'makecontext' returns.  In this case we have to
   install the context listed in the uc_link element of the context
   'makecontext' manipulated at the time of the 'makecontext' call.
   If the pointer is NULL the process must terminate.  */

ENTRY(__start_context)
	brz,pn	%i0, 1f
	 mov	1, %o1
	call	__setcontext
	 mov	%i0, %o0
1:	call	HIDDEN_JUMPTARGET(exit)
	 mov	0, %o0
	unimp	0
END(__start_context)
