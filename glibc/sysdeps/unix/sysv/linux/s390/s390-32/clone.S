/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com).

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

/* clone is even more special than fork as it mucks with stacks
   and invokes a function in the right context after its all over.  */

#include <sysdep.h>
#include <tls.h>
#define _ERRNO_H	1
#include <bits/errno.h>

/* int __clone(int (*fn)(void *arg), void *child_stack, int flags, void *arg,
               pid_t *parent_tid, void *tls, pid_t *child_tid);  */
/* sys_clone  (void *child_stack, unsigned long flags,
               pid_t *parent_tid, pid_t *child_tid, void *tls);  */

	.text
ENTRY(__clone)
	st	%r6,24(%r15)		/* store %r6 to save area */
	cfi_offset (%r6, -72)
	ltr     %r1,%r2			/* check fn and move to %r1 */
	jz      error			/* no NULL function pointers */
	lhi     %r0,-8			/* Align the child_stack to a ...  */
	nr      %r3,%r0			/* double word boundary and ...  */
	jz	error			/* avoid NULL stack pointers.  */
	lr	%r0,%r5			/* move *arg out of the way */
	lr	%r2,%r3			/* move child_stack to %r2 */
	lr	%r3,%r4			/* move flags to %r3 */
	lr	%r4,%r6			/* move parent_tid to %r4 */
	l	%r5,100(%r15)		/* load child_tid from stack */
	l	%r6,96(%r15)		/* load tls from stack */
	svc	SYS_ify(clone)
	ltr	%r2,%r2			/* check return code */
	jz	thread_start
	l	%r6,24(%r15)		/* restore %r6 */
	jm	SYSCALL_ERROR_LABEL
	br	%r14
error:
	lhi	%r2,-EINVAL
	j	SYSCALL_ERROR_LABEL
PSEUDO_END (__clone)

thread_start:
	cfi_startproc
	/* Mark r14 as undefined in order to stop unwinding here!  */
	cfi_undefined (r14)
	/* fn is in gpr 1, arg in gpr 0 */
	lr      %r2,%r0         /* set first parameter to void *arg */
	ahi     %r15,-96        /* make room on the stack for the save area */
	xc	0(4,%r15),0(%r15)
	basr    %r14,%r1        /* jump to fn */
	DO_CALL (exit, 1)
	cfi_endproc

libc_hidden_def (__clone)
weak_alias (__clone, clone)
