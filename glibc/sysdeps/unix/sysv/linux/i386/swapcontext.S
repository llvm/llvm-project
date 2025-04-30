/* Save current context and install the given one.
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


ENTRY(__swapcontext)
	/* Load address of the context data structure we save in.  */
	movl	4(%esp), %eax

	/* Save the preserved register values and the return address.  */
	movl	%edi, oEDI(%eax)
	movl	%esi, oESI(%eax)
	movl	%ebp, oEBP(%eax)
	movl	(%esp), %ecx
	movl	%ecx, oEIP(%eax)
	leal	4(%esp), %ecx
	movl	%ecx, oESP(%eax)
	movl	%ebx, oEBX(%eax)

	/* Save the FS segment register.  */
	xorl	%edx, %edx
	movw	%fs, %dx
	movl	%edx, oFS(%eax)

	/* We have separate floating-point register content memory on the
	   stack.  We use the __fpregs_mem block in the context.  Set the
	   links up correctly.  */
	leal	oFPREGSMEM(%eax), %ecx
	movl	%ecx, oFPREGS(%eax)
	/* Save the floating-point context.  */
	fnstenv	(%ecx)

	/* Load address of the context data structure we have to load.  */
	movl	8(%esp), %ecx

	/* Save the current signal mask and install the new one.  */
	pushl	%ebx
	leal	oSIGMASK(%eax), %edx
	leal	oSIGMASK(%ecx), %ecx
	movl	$SIG_SETMASK, %ebx
	movl	$__NR_sigprocmask, %eax
	ENTER_KERNEL
	popl	%ebx
	cmpl	$-4095, %eax		/* Check %eax for error.  */
	jae	SYSCALL_ERROR_LABEL	/* Jump to error handler if error.  */

	/* EAX was modified, reload it.  */
	movl	8(%esp), %eax

	/* Restore the floating-point context.  Not the registers, only the
	   rest.  */
	movl	oFPREGS(%eax), %ecx
	fldenv	(%ecx)

	/* Restore the FS segment register.  We don't touch the GS register
	   since it is used for threads.  */
	movl	oFS(%eax), %edx
	movw	%dx, %fs

#if SHSTK_ENABLED
	/* Check if Shadow Stack is enabled.  */
	testl	$X86_FEATURE_1_SHSTK, %gs:FEATURE_1_OFFSET
	jz	L(no_shstk)

	xorl	%eax, %eax
	cmpl	%gs:SSP_BASE_OFFSET, %eax
	jnz	L(shadow_stack_bound_recorded)

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
	/* Record the base of the current shadow stack.  */
	movl	8(%esp), %eax
	movl	%eax, %gs:SSP_BASE_OFFSET
	add	$24, %esp

L(shadow_stack_bound_recorded):
	/* Load address of the context data structure we save in.  */
	movl	4(%esp), %eax

	/* Load address of the context data structure we swap in  */
	movl	8(%esp), %edx

       /* If we unwind the stack, we can't undo stack unwinding.  Just
	   save the target shadow stack pointer as the current shadow
	   stack pointer.   */
	movl	oSSP(%edx), %ecx
	movl	%ecx, oSSP(%eax)

	/* Save the current shadow stack base in ucontext.  */
	movl	%gs:SSP_BASE_OFFSET, %ecx
	movl	%ecx, (oSSP + 4)(%eax)

	/* If the base of the target shadow stack is the same as the
	   base of the current shadow stack, we unwind the shadow
	   stack.  Otherwise it is a stack switch and we look for a
	   restore token.  */
	movl	oSSP(%edx), %esi
	movl	%esi, %edi

	/* Get the base of the target shadow stack.  */
	movl	(oSSP + 4)(%edx), %ecx
	cmpl	%gs:SSP_BASE_OFFSET, %ecx
	je	L(unwind_shadow_stack)

	/* Align the saved original shadow stack pointer to the next
	   8 byte aligned boundary.  */
	andl	$-8, %esi

L(find_restore_token_loop):
	/* Look for a restore token.  */
	movl	-8(%esi), %ebx
	andl	$-8, %ebx
	cmpl	%esi, %ebx
	je	L(restore_shadow_stack)

	/* Try the next slot.  */
	subl	$8, %esi
	jmp	L(find_restore_token_loop)

L(restore_shadow_stack):
	/* The target shadow stack will be restored.  Save the current
	   shadow stack pointer.  */
	rdsspd	%ecx
	movl	%ecx, oSSP(%eax)

	/* Use the restore stoken to restore the target shadow stack.  */
	rstorssp -8(%esi)

	/* Save the restore token on the old shadow stack.  NB: This
	   restore token may be checked by setcontext or swapcontext
	   later.  */
	saveprevssp

	/* Record the new shadow stack base that was switched to.  */
	movl	(oSSP + 4)(%edx), %ebx
	movl	%ebx, %gs:SSP_BASE_OFFSET

L(unwind_shadow_stack):
	rdsspd	%ebx
	subl	%edi, %ebx
	je	L(skip_unwind_shadow_stack)
	negl	%ebx
	shrl	$2, %ebx
	movl	$255, %esi
L(loop):
	cmpl	%esi, %ebx
	cmovb	%ebx, %esi
	incsspd	%esi
	subl	%esi, %ebx
	ja	L(loop)

L(skip_unwind_shadow_stack):

	/* Load the new stack pointer.  */
	movl	oESP(%edx), %esp

	/* Load the values of all the preserved registers (except ESP).  */
	movl	oEDI(%edx), %edi
	movl	oESI(%edx), %esi
	movl	oEBP(%edx), %ebp
	movl	oEBX(%edx), %ebx

	/* Get the return address set with getcontext.  */
	movl	oEIP(%edx), %ecx

	/* Check if return address is valid for the case when setcontext
	   is invoked from L(exitcode) with linked context.  */
	rdsspd	%eax
	cmpl	(%eax), %ecx
	/* Clear EAX to indicate success.  NB: Don't use xorl to keep
	   EFLAGS for jne.  */
	movl	$0, %eax
	jne	L(jmp)
	/* Return to the new context if return address valid.  */
	pushl	%ecx
	ret

L(jmp):
	/* Jump to the new context directly.  */
	jmp	*%ecx

L(no_shstk):
#endif

	/* Fetch the address to return to.  */
	movl	oEIP(%eax), %ecx

	/* Load the new stack pointer.  */
	movl	oESP(%eax), %esp

	/* Push the return address on the new stack so we can return there.  */
	pushl	%ecx

	/* Load the values of all the preserved registers (except ESP).  */
	movl	oEDI(%eax), %edi
	movl	oESI(%eax), %esi
	movl	oEBP(%eax), %ebp
	movl	oEBX(%eax), %ebx

	/* All done, return 0 for success.  */
	xorl	%eax, %eax

	/* The following 'ret' will pop the address of the code and jump
	   to it.  */
	ret
PSEUDO_END(__swapcontext)

weak_alias (__swapcontext, swapcontext)
