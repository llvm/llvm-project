/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Huggins-Daines <dhd@debian.org>, 2000.
   Based on the Alpha version by Richard Henderson <rth@tamu.edu>, 1996.

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

/* clone() is even more special than fork() as it mucks with stacks
   and invokes a function in the right context after its all over.  */

#include <sysdep.h>
#define _ERRNO_H	1
#include <bits/errno.h>
#include <tcb-offsets.h>

/* Non-thread code calls __clone with the following parameters:
   int clone(int (*fn)(void *arg),
	     void *child_stack,
	     int flags,
	     void *arg)

   NPTL Code will call __clone with the following parameters:
   int clone(int (*fn)(void *arg),
	     void *child_stack,
	     int flags,
	     void *arg,
	     int *parent_tidptr,
	     struct user_desc *newtls,
	     int *child_pidptr)

   The code should not mangle the extra input registers.
   Syscall expects:				Input to __clone:
	4(r25) - function pointer 		(r26, arg0)
	0(r25) - argument			(r23, arg3)
	r26 - clone flags.			(r24, arg2)
	r25+64 - user stack pointer.		(r25, arg1)
	r24 - parent tid pointer.		(stack - 52)
	r23 - struct user_desc newtls pointer.	(stack - 56)
	r22 - child tid pointer.		(stack - 60)
	r20 - clone syscall number		(constant)

   Return:

	On success the thread ID of the child process is returend in
	the callers context.
	On error return -1, and set errno to the value returned by
	the syscall.
 */

        .text
ENTRY(__clone)
	/* Prologue */
	stwm	%r4, 64(%sp)
	.cfi_def_cfa_offset -64
	.cfi_offset 4, 0
	stw	%sp, -4(%sp)
#ifdef PIC
	stw	%r19, -32(%sp)
	.cfi_offset 19, 32
#endif

	/* Sanity check arguments.  */
	comib,=,n	0,%arg0,.LerrorSanity	/* no NULL function pointers */
	comib,=,n	0,%arg1,.LerrorSanity	/* no NULL stack pointers */

	/* Ensure stack argument is 8-byte aligned.  */
	ldo		7(%r25),%r25
	depi		0,31,3,%r25

	/* Save the function pointer, arg, and flags on the new stack.  */
	stwm    %r26, 64(%r25)
	stw	%r23, -60(%r25)
	stw     %r24, -56(%r25)

	/* Clone arguments are (int flags, void * child_stack) */
	copy	%r24, %r26		/* flags are first */
	/* User stack pointer is in the correct register already */

	/* Load args from stack... */
	ldw	-116(%sp), %r24		/* Load parent_tidptr */
	ldw	-120(%sp), %r23 	/* Load newtls */
	ldw	-124(%sp), %r22		/* Load child_tidptr */

	/* Save the PIC register. */
#ifdef PIC
	copy	%r19, %r4		/* parent */
#endif

	/* Do the system call */
	ble     0x100(%sr2, %r0)
	ldi	__NR_clone, %r20

	ldi	-4096, %r1
	comclr,>>= %r1, %ret0, %r0	/* Note: unsigned compare. */
	b,n	.LerrorRest

	/* Restore the PIC register.  */
#ifdef PIC
	copy	%r4, %r19		/* parent */
#endif

	comib,=,n 0, %ret0, .LthreadStart

	/* Successful return from the parent
	   No need to restore the PIC register,
	   since we return immediately. */

	ldw	-84(%sp), %rp
	bv	%r0(%rp)
	ldwm	-64(%sp), %r4

.LerrorRest:
	/* Something bad happened -- no child created */
	bl	__syscall_error, %rp
	sub     %r0, %ret0, %arg0
	ldw	-84(%sp), %rp
	/* Return after setting errno, ret0 is set to -1 by __syscall_error. */
	bv	%r0(%rp)
	ldwm	-64(%sp), %r4

.LerrorSanity:
	/* Sanity checks failed, return -1, and set errno to EINVAL. */
	bl	__syscall_error, %rp
	ldi     EINVAL, %arg0
	ldw	-84(%sp), %rp
	bv	%r0(%rp)
	ldwm	-64(%sp), %r4

.LthreadStart:
	/* Load up the arguments.  */
	ldw	-60(%sp), %arg0
	ldw     -64(%sp), %r22

	/* $$dyncall fixes child's PIC register */

	/* Call the user's function */
#ifdef PIC
	copy	%r19, %r4
#endif
	bl	$$dyncall, %r31
	copy	%r31, %rp
#ifdef PIC
	copy	%r4, %r19
#endif
	copy	%r28, %r26
	ble     0x100(%sr2, %r0)
	ldi	__NR_exit, %r20

	/* We should not return from exit.
           We do not restore r4, or the stack state.  */
	iitlbp	%r0, (%sr0, %r0)

PSEUDO_END(__clone)

libc_hidden_def (__clone)
weak_alias (__clone, clone)
