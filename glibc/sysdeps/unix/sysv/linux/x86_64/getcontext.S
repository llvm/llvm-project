/* Save current context.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 2002.

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

/*  int __getcontext (ucontext_t *ucp)

  Saves the machine context in UCP such that when it is activated,
  it appears as if __getcontext() returned again.

  This implementation is intended to be used for *synchronous* context
  switches only.  Therefore, it does not have to save anything
  other than the PRESERVED state.  */


ENTRY(__getcontext)
	/* Save the preserved registers, the registers used for passing
	   args, and the return address.  */
	movq	%rbx, oRBX(%rdi)
	movq	%rbp, oRBP(%rdi)
	movq	%r12, oR12(%rdi)
	movq	%r13, oR13(%rdi)
	movq	%r14, oR14(%rdi)
	movq	%r15, oR15(%rdi)

	movq	%rdi, oRDI(%rdi)
	movq	%rsi, oRSI(%rdi)
	movq	%rdx, oRDX(%rdi)
	movq	%rcx, oRCX(%rdi)
	movq	%r8, oR8(%rdi)
	movq	%r9, oR9(%rdi)

	movq	(%rsp), %rcx
	movq	%rcx, oRIP(%rdi)
	leaq	8(%rsp), %rcx		/* Exclude the return address.  */
	movq	%rcx, oRSP(%rdi)

#if SHSTK_ENABLED
	/* Check if shadow stack is enabled.  */
	testl	$X86_FEATURE_1_SHSTK, %fs:FEATURE_1_OFFSET
	jz	L(no_shstk)

	/* Save RDI in RDX which won't be clobbered by syscall.  */
	movq	%rdi, %rdx

	xorl	%eax, %eax
	cmpq	%fs:SSP_BASE_OFFSET, %rax
	jnz	L(shadow_stack_bound_recorded)

	/* Get the base address and size of the default shadow stack
	   which must be the current shadow stack since nothing has
	   been recorded yet.  */
	sub	$24, %RSP_LP
	mov	%RSP_LP, %RSI_LP
	movl	$ARCH_CET_STATUS, %edi
	movl	$__NR_arch_prctl, %eax
	syscall
	testq	%rax, %rax
	jz	L(continue_no_err)

	/* This should never happen.  */
	hlt

L(continue_no_err):
	/* Record the base of the current shadow stack.  */
	movq	8(%rsp), %rax
	movq	%rax, %fs:SSP_BASE_OFFSET
	add	$24, %RSP_LP

	/* Restore RDI.  */
	movq	%rdx, %rdi

L(shadow_stack_bound_recorded):
	/* Get the current shadow stack pointer.  */
	rdsspq	%rax
	/* NB: Save the caller's shadow stack so that we can jump back
	   to the caller directly.  */
	addq	$8, %rax
	movq	%rax, oSSP(%rdx)

	/* Save the current shadow stack base in ucontext.  */
	movq	%fs:SSP_BASE_OFFSET, %rax
	movq	%rax, (oSSP + 8)(%rdi)

L(no_shstk):
#endif
	/* We have separate floating-point register content memory on the
	   stack.  We use the __fpregs_mem block in the context.  Set the
	   links up correctly.  */

	leaq	oFPREGSMEM(%rdi), %rcx
	movq	%rcx, oFPREGS(%rdi)
	/* Save the floating-point environment.  */
	fnstenv	(%rcx)
	fldenv	(%rcx)
	stmxcsr oMXCSR(%rdi)

	/* Save the current signal mask with
	   rt_sigprocmask (SIG_BLOCK, NULL, set,_NSIG/8).  */
	leaq	oSIGMASK(%rdi), %rdx
	xorl	%esi,%esi
#if SIG_BLOCK == 0
	xorl	%edi, %edi
#else
	movl	$SIG_BLOCK, %edi
#endif
	movl	$_NSIG8,%r10d
	movl	$__NR_rt_sigprocmask, %eax
	syscall
	cmpq	$-4095, %rax		/* Check %rax for error.  */
	jae	SYSCALL_ERROR_LABEL	/* Jump to error handler if error.  */

	/* All done, return 0 for success.  */
	xorl	%eax, %eax
	ret
PSEUDO_END(__getcontext)

weak_alias (__getcontext, getcontext)
