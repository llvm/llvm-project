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

ENTRY(__makecontext_ret)
	/* Mark r14 as undefined in order to stop unwinding here!  */
	cfi_undefined (r14)
	basr  %r14,%r7
	ltr   %r8,%r8			/* Check whether uc_link is 0.  */
	jz    1f
	lr    %r2,%r8
	br    %r9
1:	lhi   %r2,0			/* EXIT return value.  */
#ifdef NO_HIDDEN
	/* If SHARED and NO_HIDDEN is defined, we need to setup got pointer
	   as HIDDEN_JUMPTARGET expands to JUMPTARGET which expands to
	   exit@PLT.  Otherwise HIDDEN_JUMPTARGET expands to __GI_exit if
	   SHARED is defined and to exit if SHARED is not defined.  */
	SYSCALL_PIC_SETUP
#endif
	basr	%r1,0
2:
	al	%r1,3f-2b(%r1)
	basr	%r14,%r1		/* Call exit.  */
	j	.+2			/* Trap if exit returns for some reason.  */
3:
	.long   HIDDEN_JUMPTARGET (exit) - 2b
END(__makecontext_ret)
