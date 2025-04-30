/* Copyright (C) 1996-2021 Free Software Foundation, Inc.

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

/* clone() is even more special than fork() as it mucks with stacks
   and invokes a function in the right context after its all over.  */

#include <sysdep.h>
#define _ERRNO_H	1
#include <bits/errno.h>

/* int clone(int (*fn)(void *arg),            x0
	     void *child_stack,               x1
	     int flags,                       x2
	     void *arg,                       x3
	     pid_t *ptid,                     x4
	     struct user_desc *tls,           x5
             pid_t *ctid);                    x6
 */
        .text
ENTRY(__clone)
	PTR_ARG (0)
	PTR_ARG (1)
	PTR_ARG (3)
	PTR_ARG (4)
	PTR_ARG (5)
	PTR_ARG (6)
	/* Save args for the child.  */
	mov	x10, x0
	mov	x11, x2
	mov	x12, x3

	/* Sanity check args.  */
	mov	x0, #-EINVAL
	cbz	x10, .Lsyscall_error
	/* Align sp.  */
	and	x1, x1, -16
	cbz	x1, .Lsyscall_error

	/* Do the system call.  */
	/* X0:flags, x1:newsp, x2:parenttidptr, x3:newtls, x4:childtid.  */
	mov	x0, x2                  /* flags  */
	/* New sp is already in x1.  */
	mov	x2, x4			/* ptid  */
	mov	x3, x5			/* tls  */
	mov	x4, x6			/* ctid  */
	mov	x8, #SYS_ify(clone)
	svc	0x0

	cmp	x0, #0
	beq	thread_start
	blt	.Lsyscall_error
	RET
PSEUDO_END (__clone)

	.align 4
	.type thread_start, %function
thread_start:
	cfi_startproc
	cfi_undefined (x30)
	mov	x29, 0

	/* Pick the function arg and execute.  */
	mov	x0, x12
	blr	x10

	/* We are done, pass the return value through x0.  */
	mov	x8, #SYS_ify(exit)
	svc	0x0
	cfi_endproc
	.size thread_start, .-thread_start

libc_hidden_def (__clone)
weak_alias (__clone, clone)
