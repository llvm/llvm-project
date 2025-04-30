/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
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
#include <asm-syntax.h>

/* The userland implementation is:
   int clone (int (*fn)(void *arg), void *child_stack, int flags,
	      void *arg, pid_t *parent_tid, void *tls, pid_t *child_tid);
   the kernel entry is:
   int clone (long flags, void *child_stack, pid_t *parent_tid,
	      pid_t *child_tid, void *tls);

   The parameters are passed in register and on the stack from userland:
   rdi: fn
   rsi: child_stack
   rdx:	flags
   rcx: arg
    r8:	TID field in parent
    r9: thread pointer
%rsp+8:	TID field in child

   The kernel expects:
   rax: system call number
   rdi: flags
   rsi: child_stack
   rdx: TID field in parent
   r10: TID field in child
   r8:	thread pointer  */


        .text
ENTRY (__clone)
	/* Sanity check arguments.  */
	movq	$-EINVAL,%rax
	testq	%rdi,%rdi		/* no NULL function pointers */
	jz	SYSCALL_ERROR_LABEL

	/* Align stack to 16 bytes per the x86-64 psABI.  */
	andq	$-16, %rsi
	jz	SYSCALL_ERROR_LABEL	/* no NULL stack pointers */

	/* Insert the argument onto the new stack.  */
	movq	%rcx,-8(%rsi)

	subq	$16,%rsi

	/* Save the function pointer.  It will be popped off in the
	   child.  */
	movq	%rdi,0(%rsi)

	/* Do the system call.  */
	movq	%rdx, %rdi
	movq	%r8, %rdx
	movq	%r9, %r8
	mov	8(%rsp), %R10_LP
	movl	$SYS_ify(clone),%eax

	/* End FDE now, because in the child the unwind info will be
	   wrong.  */
	cfi_endproc;
	syscall

	testq	%rax,%rax
	jl	SYSCALL_ERROR_LABEL
	jz	L(thread_start)

	ret

L(thread_start):
	cfi_startproc;
	/* Clearing frame pointer is insufficient, use CFI.  */
	cfi_undefined (rip);
	/* Clear the frame pointer.  The ABI suggests this be done, to mark
	   the outermost frame obviously.  */
	xorl	%ebp, %ebp

	/* Set up arguments for the function call.  */
	popq	%rax		/* Function to call.  */
	popq	%rdi		/* Argument.  */
	call	*%rax
	/* Call exit with return value from function call. */
	movq	%rax, %rdi
	movl	$SYS_ify(exit), %eax
	syscall
	cfi_endproc;

	cfi_startproc;
PSEUDO_END (__clone)

libc_hidden_def (__clone)
weak_alias (__clone, clone)
