/* _Unwind_Resume wrapper for ARM EABI.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

/* This is equivalent to the following C implementation:

   void
   _Unwind_Resume (struct _Unwind_Exception *exc)
   {
     __unwind_link_get_resume () (exc);
   }

   We do it in assembly just to ensure that we get an unmolested tail
   call to the libgcc function, which is necessary for the ARM unwinder.  */

ENTRY (_Unwind_Resume)
	/* We need to save and restore LR (for our own return address)
	   and R0 (for the argument to _Unwind_Resume) around the call.  */
	push	{r0, lr}
	cfi_adjust_cfa_offset (8)
	cfi_rel_offset (r0, 0)
	cfi_rel_offset (lr, 4)
	bl	__unwind_link_get_resume
	mov	r3, r0
	pop	{r0, lr}
	cfi_adjust_cfa_offset (-8)
	cfi_restore (r4)
	cfi_restore (lr)
	bx	r3
END (_Unwind_Resume)
