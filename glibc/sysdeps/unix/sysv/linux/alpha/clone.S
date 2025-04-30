/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson <rth@tamu.edu>, 1996.

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

/* int clone(int (*fn)(void *arg), void *child_stack, int flags,
	     void *arg, pid_t *ptid, void *tls, pid_t *ctid);

   Note that everything past ARG is technically optional, based
   on FLAGS, and that CTID is arg 7, and thus is on the stack.
   However, since a load from top-of-stack better be legal always,
   we don't bother checking FLAGS.  */

        .text
	.align	4
	.globl	__clone
	.ent	__clone
	.usepv	__clone, USEPV_PROF

	cfi_startproc
__clone:
#ifdef PROF
	.set noat
	ldgp	gp,0(pv)
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.set at
#endif

	/* Sanity check arguments.  */
	ldiq	v0, EINVAL
	beq	a0, SYSCALL_ERROR_LABEL	/* no NULL function pointers */
	beq	a1, SYSCALL_ERROR_LABEL	/* no NULL stack pointers */

	/* Save the fn ptr and arg on the new stack.  */
	subq	a1, 32, a1
	stq	a0, 0(a1)
	stq	a3, 8(a1)
	stq	a2, 16(a1)

	/* The syscall is of the form clone(flags, usp, ptid, ctid, tls).
	   Shift the flags, ptid, ctid, tls arguments into place; the
	   child_stack argument is already correct.  */
	mov	a2, a0
	mov	a4, a2
	ldq	a3, 0(sp)
	mov	a5, a4

	/* Do the system call.  */
	ldiq	v0, __NR_clone
	call_pal PAL_callsys

	bne	a3, SYSCALL_ERROR_LABEL
	beq	v0, thread_start

	/* Successful return from the parent.  */
	ret

PSEUDO_END(__clone)
	cfi_endproc

/* Load up the arguments to the function.  Put this block of code in
   its own function so that we can terminate the stack trace with our
   debug info.  */

	.align	4
	.ent thread_start
	cfi_startproc
thread_start:
	mov	0, fp
	cfi_def_cfa_register(fp)
	cfi_undefined(ra)

	/* Load up the arguments.  */
	ldq	pv, 0(sp)
	ldq	a0, 8(sp)
	addq	sp, 32, sp

	/* Call the user's function.  */
	jsr	ra, (pv)
	ldgp	gp, 0(ra)

	mov     v0, a0
	ldiq	v0, __NR_exit
	call_pal PAL_callsys

	/* Die horribly.  */
	.align	4
	halt

	.align	4
	cfi_endproc
	.end thread_start

libc_hidden_def (__clone)
weak_alias (__clone, clone)
