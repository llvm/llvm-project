/* clone() implementation for Nios II.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andrew Jenner <andrew@codesourcery.com>, 2008.

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
#include <tcb-offsets.h>

/* int clone(int (*fn)(void *arg), void *child_stack, int flags, void *arg,
	     void *parent_tidptr, void *tls, void *child_tidptr) */

	.text
ENTRY(__clone)
	/* Sanity check arguments.  */
	movi	r2, EINVAL
	/* No NULL function pointers.  */
	beq	r4, zero, SYSCALL_ERROR_LABEL
	/* No NULL stack pointers.  */
	beq	r5, zero, SYSCALL_ERROR_LABEL

	subi	r5, r5, 8	/* Reserve argument save space.  */
	stw	r4, 4(r5)	/* Save function pointer.  */
	stw	r7, 0(r5)	/* Save argument pointer.  */

	/* Load arguments.  */
	mov	r4, r6
	ldw	r6, 0(sp)
	ldw	r7, 8(sp)
	ldw	r8, 4(sp)

	/* Do the system call.  */
	movi	r2, SYS_ify (clone)

	/* End FDE now, because in the child the unwind info will be
	   wrong.  */
	cfi_endproc
	trap

	/* Check for errors.  */
	bne	r7, zero, SYSCALL_ERROR_LABEL
	/* See if we're on the newly created thread.  */
	beq	r2, zero, thread_start
	/* Successful return from the parent */
	ret

thread_start:
	cfi_startproc
	cfi_undefined (ra)

	ldw	r5, 4(sp)	/* Function pointer.  */
	ldw	r4, 0(sp)	/* Argument pointer.  */
	addi	sp, sp, 8

        /* Call the user's function.  */
	callr	r5

	/* exit with the result.  */
	movi	r2, SYS_ify (exit)
	trap
	cfi_endproc

	cfi_startproc
PSEUDO_END (__clone)
libc_hidden_def (__clone)
weak_alias (__clone, clone)
