/* The clone3 syscall wrapper.  Linux/i386 version.
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

   The parameters are passed on stack from userland:
   16(%esp)	arg
   12(%esp)	func
    8(%esp)	size
    4(%esp)	cl_args
     (%esp)	Return address

   The kernel expects:
   eax:		system call number
   ebx:		cl_args
   ecx:		size
 */

#define CL_ARGS	4
#define SIZE	8
#define FUNC	12
#define ARG	16

        .text
ENTRY (__clone3)
	/* Sanity check arguments.  */
	movl	$-EINVAL, %eax
	movl	CL_ARGS(%esp), %ecx	/* No NULL cl_args pointer.  */
	testl	%ecx, %ecx
	jz	SYSCALL_ERROR_LABEL
	/* Save the function pointer in EDX which is preserved by the
	   system call.  */
	movl	FUNC(%esp), %edx	/* No NULL function pointer.  */
	testl	%edx, %edx
	jz	SYSCALL_ERROR_LABEL

	/* Save EBX and ESI.  */
	pushl	%ebx
	cfi_adjust_cfa_offset (4)
	pushl	%esi
	cfi_adjust_cfa_offset (4)

	/* Save the function argument in ESI which is preserved by the
	   system call.  */
	movl	(ARG + 8)(%esp), %esi

	/* Put cl_args in EBX.  */
	movl	%ecx, %ebx

	/* Put size in ECX.  */
	movl	(SIZE + 8)(%esp), %ecx

	/* Do the system call.  */
	movl	$SYS_ify(clone3), %eax

	/* End FDE now, because in the child the unwind info will be
	   wrong.  */
	cfi_endproc

	int	$0x80
	test	%eax, %eax
	/* No need to restore EBX and ESI in child.  */
	jz	L(thread_start)

	/* Restore EBX and ESI in parent.  */
	pop	%esi
	pop	%ebx
	jl	SYSCALL_ERROR_LABEL

	ret

L(thread_start):
	cfi_startproc
	/* Clearing frame pointer is insufficient, use CFI.  */
	cfi_undefined (eip)
	xorl	%ebp, %ebp	/* Terminate the stack frame.  */

	/* Align stack to 16 bytes per the i386 psABI.  */
	andl	$-16, %esp

	/* The PUSH below will decrement stack pointer by 4 bytes.  */
	subl	$12, %esp

	/* Set up the argument for the function call.  */
	pushl	%esi		/* Argument.  */
	cfi_adjust_cfa_offset (4)
	call	*%edx		/* Call function.  */

	/* Call exit with return value from function call. */
	movl	%eax, %ebx
	movl	$SYS_ify(exit), %eax
	ENTER_KERNEL
	cfi_endproc

	cfi_startproc
PSEUDO_END (__clone3)

libc_hidden_def (__clone3)
weak_alias (__clone3, clone3)
