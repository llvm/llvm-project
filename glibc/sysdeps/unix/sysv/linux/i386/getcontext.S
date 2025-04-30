/* Save current context.
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
#include <asm/prctl.h>

#include "ucontext_i.h"


ENTRY(__getcontext)
	/* Load address of the context data structure.  */
	movl	4(%esp), %eax

	/* Save the preserved register values and the return address.  */
	movl	%edi, oEDI(%eax)
	movl	%esi, oESI(%eax)
	movl	%ebp, oEBP(%eax)
	movl	(%esp), %ecx
	movl	%ecx, oEIP(%eax)
	leal	4(%esp), %ecx		/* Exclude the return address.  */
	movl	%ecx, oESP(%eax)
	movl	%ebx, oEBX(%eax)

	/* Save the FS segment register.  We don't touch the GS register
	   since it is used for threads.  */
	xorl	%edx, %edx
	movw	%fs, %dx
	movl	%edx, oFS(%eax)

#if SHSTK_ENABLED
	/* Check if shadow stack is enabled.  */
	testl	$X86_FEATURE_1_SHSTK, %gs:FEATURE_1_OFFSET
	jz	L(no_shstk)

	/* Save EAX in EDX.  */
	movl	%eax, %edx

	xorl	%eax, %eax
	cmpl	%gs:SSP_BASE_OFFSET, %eax
	jnz	L(shadow_stack_bound_recorded)

	/* Save EBX in the first scratch register slot.  */
	movl	%ebx, oSCRATCH1(%edx)

	/* Get the base address and size of the default shadow stack
	   which must be the current shadow stack since nothing has
	   been recorded yet.  */
	sub	$24, %esp
	mov	%esp, %ecx
	movl	$ARCH_CET_STATUS, %ebx
	movl	$__NR_arch_prctl, %eax
	ENTER_KERNEL
	testl	%eax, %eax
	jz	L(continue_no_err)

	/* This should never happen.  */
	hlt

L(continue_no_err):
	/* Restore EBX from the first scratch register slot.  */
	movl	oSCRATCH1(%edx), %ebx

	/* Record the base of the current shadow stack.  */
	movl	8(%esp), %eax
	movl	%eax, %gs:SSP_BASE_OFFSET
	add	$24, %esp

L(shadow_stack_bound_recorded):
	/* Load address of the context data structure.  */
	movl	4(%esp), %eax

	/* Get the current shadow stack pointer.  */
	rdsspd	%edx
	/* NB: Save the caller's shadow stack so that we can jump back
	   to the caller directly.  */
	addl	$4, %edx
	movl	%edx, oSSP(%eax)

	/* Save the current shadow stack base in ucontext.  */
	movl	%gs:SSP_BASE_OFFSET, %edx
	movl	%edx, (oSSP + 4)(%eax)

L(no_shstk):
#endif
	/* We have separate floating-point register content memory on the
	   stack.  We use the __fpregs_mem block in the context.  Set the
	   links up correctly.  */
	leal	oFPREGSMEM(%eax), %ecx
	movl	%ecx, oFPREGS(%eax)
	/* Save the floating-point context.  */
	fnstenv	(%ecx)
	/* And load it right back since the processor changes the mask.
	   Intel thought this opcode to be used in interrupt handlers which
	   would block all exceptions.  */
	fldenv	(%ecx)

	/* Save the current signal mask.  */
	pushl	%ebx
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (ebx, 0)
	leal	oSIGMASK(%eax), %edx
	xorl	%ecx, %ecx
	movl	$SIG_BLOCK, %ebx
	movl	$__NR_sigprocmask, %eax
	ENTER_KERNEL
	popl	%ebx
	cfi_adjust_cfa_offset (-4)
	cfi_restore (ebx)
	cmpl	$-4095, %eax		/* Check %eax for error.  */
	jae	SYSCALL_ERROR_LABEL	/* Jump to error handler if error.  */

	/* All done, return 0 for success.  */
	xorl	%eax, %eax
	ret
PSEUDO_END(__getcontext)

weak_alias (__getcontext, getcontext)
