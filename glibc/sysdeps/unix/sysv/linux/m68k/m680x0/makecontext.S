/* Create new context.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include "ucontext_i.h"


ENTRY(__makecontext)
	move.l	4(%sp), %a0

	/* Get the address of the function we are supposed to run.  */
	move.l	8(%sp), oPC(%a0)

	/* Compute the address of the stack.  The information comes from
	   to us_stack element.  */
	move.l	oSS_SP(%a0), %a1
	add.l	oSS_SIZE(%a0), %a1

	/* Remember the number of parameters for the exit handler since
	   it has to remove them.  We store the number in the D7 register
	   which the function we will call must preserve.  */
	move.l	12(%sp), %d1
	move.l	%d1, oGREGS+7*4(%a0)

	/* Make room on the new stack for the parameters.
	   Room for the arguments, return address (== 1f) and
	   oLINK pointer is needed.  */
	neg.l	%d1
	lea	-8(%a1,%d1.l*4), %a1
	neg.l	%d1

	/* Store the future stack pointer.  */
	move.l	%a1, oSP(%a0)

	/* Put the next context on the new stack (from the uc_link
	   element).  */
	move.l	oLINK(%a0), 4(%a1,%d1.l*4)

	/* Copy all the parameters.  */
1:	subq.l	#1,%d1
	jmi	2f
	move.l	16(%sp,%d1.l*4), 4(%a1,%d1.l*4)
	jra	1b
2:

	/* If the function we call returns we must continue with the
	   context which is given in the uc_link element.  To do this
	   set the return address for the function the user provides
	   to a little bit of helper code which does the magic (see
	   below).  */
	lea	1f(%pc), %a0
	move.l	%a0, (%a1)
	/* 'makecontext' returns no value.  */
	rts

	/* This is the helper code which gets called if a function which
	   is registered with 'makecontext' returns.  In this case we
	   have to install the context listed in the uc_link element of
	   the context 'makecontext' manipulated at the time of the
	   'makecontext' call.  If the pointer is NULL the process must
	   terminate.
	   Make sure to separate the return label from the previous unwind
	   region, because the unwinder uses ra-1 to find the FDE.  */
	cfi_endproc
	nop
1:
	/* This removes the parameters passed to the function given to
	   'makecontext' from the stack.  D7 contains the number of
	   parameters (see above).  */
	lea	(%sp,%d7.l*4), %sp

	tst.l	(%sp)			/* Check the next context.  */
	jeq	2f			/* If it is zero exit.  */

	jbsr	JUMPTARGET(__setcontext)
	/* If this returns (which can happen if the syscall fails) we'll
	   exit the program with the return error value (-1).  */

	move.l	%d0, (%sp)
2:	jbsr	HIDDEN_JUMPTARGET(exit)
	/* The 'exit' call should never return.  In case it does cause
	   the process to terminate.  */
	illegal
	cfi_startproc
END(__makecontext)

weak_alias (__makecontext, makecontext)
