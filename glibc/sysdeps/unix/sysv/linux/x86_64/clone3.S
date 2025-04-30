/* The clone3 syscall wrapper.  Linux/x86-64 version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

/* clone3() is even more special than fork() as it mucks with stacks
   and invokes a function in the right context after its all over.  */

#include <sysdep.h>

/* The userland implementation is:
   int clone3 (struct clone_args *cl_args, size_t size,
	       int (*func)(void *arg), void *arg);
   the kernel entry is:
   int clone3 (struct clone_args *cl_args, size_t size);

   The parameters are passed in registers from userland:
   rdi: cl_args
   rsi: size
   rdx: func
   rcx: arg

   The kernel expects:
   rax: system call number
   rdi: cl_args
   rsi: size  */

        .text
ENTRY (__clone3)
	/* Sanity check arguments.  */
	movl	$-EINVAL, %eax
	test	%RDI_LP, %RDI_LP	/* No NULL cl_args pointer.  */
	jz	SYSCALL_ERROR_LABEL
	test	%RDX_LP, %RDX_LP	/* No NULL function pointer.  */
	jz	SYSCALL_ERROR_LABEL

	/* Save the cl_args pointer in R8 which is preserved by the
	   syscall.  */
	mov	%RCX_LP, %R8_LP

	/* Do the system call.  */
	movl	$SYS_ify(clone3), %eax

	/* End FDE now, because in the child the unwind info will be
	   wrong.  */
	cfi_endproc
	syscall

	test	%RAX_LP, %RAX_LP
	jl	SYSCALL_ERROR_LABEL
	jz	L(thread_start)

	ret

L(thread_start):
	cfi_startproc
	/* Clearing frame pointer is insufficient, use CFI.  */
	cfi_undefined (rip)
	/* Clear the frame pointer.  The ABI suggests this be done, to mark
	   the outermost frame obviously.  */
	xorl	%ebp, %ebp

	/* Align stack to 16 bytes per the x86-64 psABI.  */
	and	$-16, %RSP_LP

	/* Set up arguments for the function call.  */
	mov	%R8_LP, %RDI_LP	/* Argument.  */
	call	*%rdx		/* Call function.  */
	/* Call exit with return value from function call. */
	movq	%rax, %rdi
	movl	$SYS_ify(exit), %eax
	syscall
	cfi_endproc

	cfi_startproc
PSEUDO_END (__clone3)

libc_hidden_def (__clone3)
weak_alias (__clone3, clone3)
