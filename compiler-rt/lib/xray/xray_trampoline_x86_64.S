//===-- xray_trampoline_x86.s -----------------------------------*- ASM -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// This implements the X86-specific assembler for the trampolines.
//
//===----------------------------------------------------------------------===//

	.text
	.file "xray_trampoline_x86.S"
	.globl __xray_FunctionEntry
	.align 16, 0x90
	.type __xray_FunctionEntry,@function

__xray_FunctionEntry:
  .cfi_startproc
  // Save caller provided registers before doing any actual work.
	pushq %rbp
	.cfi_def_cfa_offset 16
	subq $200, %rsp
	movupd	%xmm0, 184(%rsp)
	movupd	%xmm1, 168(%rsp)
	movupd	%xmm2, 152(%rsp)
	movupd	%xmm3, 136(%rsp)
	movupd	%xmm4, 120(%rsp)
	movupd	%xmm5, 104(%rsp)
	movupd	%xmm6, 88(%rsp)
	movupd	%xmm7, 72(%rsp)
	movq	%rdi, 64(%rsp)
	movq  %rax, 56(%rsp)
	movq  %rdx, 48(%rsp)
	movq	%rsi, 40(%rsp)
	movq	%rcx, 32(%rsp)
	movq	%r8, 24(%rsp)
	movq	%r9, 16(%rsp)

	// de-mangled, that's __xray::XRayPatchedFunction, and we're doing an acquire
	// load (on x86 is a normal mov instruction).
	movq	_ZN6__xray19XRayPatchedFunctionE(%rip), %rax
	testq	%rax, %rax
	je	.Ltmp0

	// assume that %r10d has the function id.
	movl	%r10d, %edi
	xor	%esi,%esi
	callq	*%rax
.Ltmp0:
  // restore the registers
	movupd	184(%rsp), %xmm0
	movupd	168(%rsp), %xmm1
	movupd	152(%rsp), %xmm2
	movupd	136(%rsp), %xmm3
	movupd	120(%rsp), %xmm4
	movupd	104(%rsp), %xmm5
	movupd	88(%rsp) , %xmm6
	movupd	72(%rsp) , %xmm7
	movq	64(%rsp), %rdi
	movq  56(%rsp), %rax
	movq  48(%rsp), %rdx
	movq	40(%rsp), %rsi
	movq	32(%rsp), %rcx
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	addq	$200, %rsp
	popq	%rbp
	retq
.Ltmp1:
	.size __xray_FunctionEntry, .Ltmp1-__xray_FunctionEntry
	.cfi_endproc

	.globl __xray_FunctionExit
	.align 16, 0x90
	.type __xray_FunctionExit,@function
__xray_FunctionExit:
	.cfi_startproc
	// Save the important registers first. Since we're assuming that this
	// function is only jumped into, we only preserve the registers for
	// returning.
	pushq	%rbp
	.cfi_def_cfa_offset 16
	subq	$56, %rsp
	.cfi_def_cfa_offset 32
	movupd	%xmm0, 40(%rsp)
	movupd	%xmm1, 24(%rsp)
	movq	%rax, 16(%rsp)
	movq	%rdx, 8(%rsp)
	movq	_ZN6__xray19XRayPatchedFunctionE(%rip), %rax
	testq %rax,%rax
	je	.Ltmp2

	movl	%r10d, %edi
	movl	$1, %esi
	callq	*%rax
.Ltmp2:
  // Restore the important registers.
	movupd	40(%rsp), %xmm0
	movupd	24(%rsp), %xmm1
	movq	16(%rsp), %rax
	movq	8(%rsp), %rdx
	addq	$56, %rsp
	popq	%rbp
	retq
.Ltmp3:
	.size __xray_FunctionExit, .Ltmp3-__xray_FunctionExit
	.cfi_endproc
