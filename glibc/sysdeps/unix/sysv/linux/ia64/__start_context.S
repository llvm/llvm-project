/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Mosberger-Tang <davidm@hpl.hp.com>.

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
#include <features.h>

/* When a context set up by __makecontext() is activated, control
   transfers to __start_context.  When we get here:

	b1 = entry point of function to call
	in0 = address of UCP to resume after function returns
	in1 = global pointer for __start_context
	out0 .. outN = arguments for function  */

ENTRY(__start_context)
	.prologue
	alloc r2 = ar.pfs, 2, 0, 8, 0

	.save rp, r4		// terminate call chain with a NULL rp
	mov r4 = r0
	;;

	.body
	br.call.sptk rp = b1
1:
	mov gp = in1			// restore gp
	cmp.ne p6,p0 = in0, r0		// uc_link != 0 ?
	;;
(p6)	mov out0 = in0
(p6)	br.call.sptk rp = __setcontext
.Lexit:
	mov out0 = 0
	br.call.sptk rp = HIDDEN_JUMPTARGET(exit)

1:	br.cond.sptk .Lexit
END(__start_context)
