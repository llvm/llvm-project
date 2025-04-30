/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson (rth@tamu.edu)

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

/* int clone(int (*fn)(void *arg), void *child_stack, int flags, void *arg,
	     pid_t *ptid, struct user_desc *tls, pid_t *ctid); */

#define PARMS	4		/* no space for saved regs */
#define FUNC	PARMS
#define STACK	FUNC+4
#define FLAGS	STACK+4
#define ARG	FLAGS+4
#define PTID	ARG+4
#define TLS	PTID+4
#define CTID	TLS+4

#define __NR_clone 120
#define SYS_clone 120

        .text
ENTRY (__clone)
	/* Sanity check arguments.  */
	movl	$-EINVAL,%eax
	movl	FUNC(%esp),%ecx		/* no NULL function pointers */
	testl	%ecx,%ecx
	jz	SYSCALL_ERROR_LABEL
	movl	STACK(%esp),%ecx	/* no NULL stack pointers */
	testl	%ecx,%ecx
	jz	SYSCALL_ERROR_LABEL

	/* Insert the argument onto the new stack.  Make sure the new
	   thread is started with an alignment of (mod 16).  */
	andl	$0xfffffff0, %ecx
	subl	$28,%ecx
	movl	ARG(%esp),%eax		/* no negative argument counts */
	movl	%eax,12(%ecx)

	/* Save the function pointer as the zeroth argument.
	   It will be popped off in the child in the ebx frobbing below.  */
	movl	FUNC(%esp),%eax
	movl	%eax,8(%ecx)
	/* Don't leak any information.  */
	movl	$0,4(%ecx)

	/* Do the system call */
	pushl	%ebx
	cfi_adjust_cfa_offset (4)
	pushl	%esi
	cfi_adjust_cfa_offset (4)
	pushl	%edi
	cfi_adjust_cfa_offset (4)

	movl	TLS+12(%esp),%esi
	cfi_rel_offset (esi, 4)
	movl	PTID+12(%esp),%edx
	movl	FLAGS+12(%esp),%ebx
	cfi_rel_offset (ebx, 8)
	movl	CTID+12(%esp),%edi
	cfi_rel_offset (edi, 0)
	movl	$SYS_ify(clone),%eax

	/* Remember the flag value.  */
	movl	%ebx, (%ecx)

	/* End FDE now, because in the child the unwind info will be
	   wrong.  */
	cfi_endproc

	int	$0x80
	popl	%edi
	popl	%esi
	popl	%ebx

	test	%eax,%eax
	jl	SYSCALL_ERROR_LABEL
	jz	L(thread_start)

	ret

L(thread_start):
	cfi_startproc;
	/* Clearing frame pointer is insufficient, use CFI.  */
	cfi_undefined (eip);
	/* Note: %esi is zero.  */
	movl	%esi,%ebp	/* terminate the stack frame */
	call	*%ebx
	movl	%eax, %ebx
	movl	$SYS_ify(exit), %eax
	ENTER_KERNEL

PSEUDO_END (__clone)

libc_hidden_def (__clone)
weak_alias (__clone, clone)
