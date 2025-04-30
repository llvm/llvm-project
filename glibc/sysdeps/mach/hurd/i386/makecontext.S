/* Create new context.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2001.

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
	movl	4(%esp), %eax

	/* Load the address of the function we are supposed to run.  */
	movl	8(%esp), %ecx

	/* Compute the address of the stack.  The information comes from
	   to us_stack element.  */
	movl	oSS_SP(%eax), %edx
	movl	%ecx, oEIP(%eax)
	addl	oSS_SIZE(%eax), %edx

	/* Remember the number of parameters for the exit handler since
	   it has to remove them.  We store the number in the EBX register
	   which the function we will call must preserve.  */
	movl	12(%esp), %ecx
	movl	%ecx, oEBX(%eax)

	/* Make room on the new stack for the parameters.
	   Room for the arguments, return address (== L(exitcode)) and
	   oLINK pointer is needed.  One of the pointer sizes is subtracted
	   after aligning the stack.  */
	negl	%ecx
	leal	-4(%edx,%ecx,4), %edx
	negl	%ecx

	/* Align the stack.  */
	andl	$0xfffffff0, %edx
	subl	$4, %edx

	/* Store the future stack pointer.  */
	movl	%edx, oESP(%eax)

	/* Put the next context on the new stack (from the uc_link
	   element).  */
	movl	oLINK(%eax), %eax
	movl	%eax, 4(%edx,%ecx,4)

	/* Copy all the parameters.  */
	jecxz	2f
1:	movl	12(%esp,%ecx,4), %eax
	movl	%eax, (%edx,%ecx,4)
	decl	%ecx
	jnz	1b
2:

	/* If the function we call returns we must continue with the
	   context which is given in the uc_link element.  To do this
	   set the return address for the function the user provides
	   to a little bit of helper code which does the magic (see
	   below).  */
#ifdef PIC
	call	1f
	cfi_adjust_cfa_offset (4)
1:	popl	%ecx
	cfi_adjust_cfa_offset (-4)
	addl	$L(exitcode)-1b, %ecx
	movl	%ecx, (%edx)
#else
	movl	$L(exitcode), (%edx)
#endif
	/* We need to terminate the FDE here instead of after ret because
	   the unwinder looks at ra-1 for unwind information.  */
	cfi_endproc

	/* 'makecontext' returns no value.  */
	ret

	/* This is the helper code which gets called if a function which
	   is registered with 'makecontext' returns.  In this case we
	   have to install the context listed in the uc_link element of
	   the context 'makecontext' manipulated at the time of the
	   'makecontext' call.  If the pointer is NULL the process must
	   terminate.  */
L(exitcode):
	/* This removes the parameters passed to the function given to
	   'makecontext' from the stack.  EBX contains the number of
	   parameters (see above).  */
	leal	(%esp,%ebx,4), %esp

	cmpl	$0, (%esp)		/* Check the next context.  */
	je	2f			/* If it is zero exit.  */

	call	HIDDEN_JUMPTARGET(__setcontext)
	/* If this returns (which can happen if the syscall fails) we'll
	   exit the program with the return error value (-1).  */
	jmp L(call_exit)

2:
	/* Exit with status 0.  */
	xorl	%eax, %eax

L(call_exit):
	/* Align the stack and pass the exit code (from %eax).  */
	andl	$0xfffffff0, %esp
	subl	$12, %esp
	pushl	%eax

	call	HIDDEN_JUMPTARGET(exit)
	/* The 'exit' call should never return.  In case it does cause
	   the process to terminate.  */
	hlt
	cfi_startproc
END(__makecontext)

weak_alias (__makecontext, makecontext)
