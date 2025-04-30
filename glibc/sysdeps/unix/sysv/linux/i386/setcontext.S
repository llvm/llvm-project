/* Install given context.
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


ENTRY(__setcontext)
	/* Load address of the context data structure.  */
	movl	4(%esp), %eax

	/* Get the current signal mask.  Note that we preserve EBX in case
	   the system call fails and we return from the function with an
	   error.  */
	pushl	%ebx
	cfi_adjust_cfa_offset (4)
	xorl	%edx, %edx
	leal	oSIGMASK(%eax), %ecx
	movl	$SIG_SETMASK, %ebx
	cfi_rel_offset (ebx, 0)
	movl	$__NR_sigprocmask, %eax
	ENTER_KERNEL
	popl	%ebx
	cfi_adjust_cfa_offset (-4)
	cfi_restore (ebx)
	cmpl	$-4095, %eax		/* Check %eax for error.  */
	jae	SYSCALL_ERROR_LABEL	/* Jump to error handler if error.  */

	/* EAX was modified, reload it.  */
	movl	4(%esp), %eax

	/* Restore the floating-point context.  Not the registers, only the
	   rest.  */
	movl	oFPREGS(%eax), %ecx
	fldenv	(%ecx)

	/* Restore the FS segment register.  We don't touch the GS register
	   since it is used for threads.  */
	movl	oFS(%eax), %ecx
	movw	%cx, %fs

	/* Load the new stack pointer.  */
	cfi_def_cfa (eax, 0)
	cfi_offset (edi, oEDI)
	cfi_offset (esi, oESI)
	cfi_offset (ebp, oEBP)
	cfi_offset (ebx, oEBX)
	movl	oESP(%eax), %esp

#if SHSTK_ENABLED
	/* Check if Shadow Stack is enabled.  */
	testl	$X86_FEATURE_1_SHSTK, %gs:FEATURE_1_OFFSET
	jz	L(no_shstk)

	/* If the base of the target shadow stack is the same as the
	   base of the current shadow stack, we unwind the shadow
	   stack.  Otherwise it is a stack switch and we look for a
	   restore token.  */
	movl	oSSP(%eax), %esi
	movl	%esi, %edi

	/* Get the base of the target shadow stack.  */
	movl	(oSSP + 4)(%eax), %ecx
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
	/* Pop return address from the shadow stack since setcontext
	   will not return.  */
	movl	$1, %ebx
	incsspd	%ebx

	/* Use the restore stoken to restore the target shadow stack.  */
	rstorssp -8(%esi)

	/* Save the restore token on the old shadow stack.  NB: This
	   restore token may be checked by setcontext or swapcontext
	   later.  */
	saveprevssp

	/* Record the new shadow stack base that was switched to.  */
	movl	(oSSP + 4)(%eax), %ebx
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

	/* Load the values of all the preserved registers (except ESP).  */
	movl	oEDI(%eax), %edi
	movl	oESI(%eax), %esi
	movl	oEBP(%eax), %ebp
	movl	oEBX(%eax), %ebx

	/* Get the return address set with getcontext.  */
	movl	oEIP(%eax), %ecx

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

	/* Push the return address on the new stack so we can return there.  */
	pushl	%ecx

	/* Load the values of all the preserved registers (except ESP).  */
	movl	oEDI(%eax), %edi
	movl	oESI(%eax), %esi
	movl	oEBP(%eax), %ebp
	movl	oEBX(%eax), %ebx

	/* All done, return 0 for success.  */
	xorl	%eax, %eax

	/* End FDE here, we fall into another context.  */
	cfi_endproc
	cfi_startproc

	/* The following 'ret' will pop the address of the code and jump
	   to it.  */

	ret
PSEUDO_END(__setcontext)
libc_hidden_def (__setcontext)

weak_alias (__setcontext, setcontext)
