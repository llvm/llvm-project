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
	subq $72, %rsp
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
	movq	64(%rsp), %rdi
	movq  56(%rsp), %rax
	movq  48(%rsp), %rdx
	movq	40(%rsp), %rsi
	movq	32(%rsp), %rcx
	movq	24(%rsp), %r8
	movq	16(%rsp), %r9
	addq	$72, %rsp
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
	// FIXME: Figure out whether this is sufficient.
	pushq	%rbp
	.cfi_def_cfa_offset 16
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
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
	movq	16(%rsp), %rax
	movq	8(%rsp), %rdx
	addq	$24, %rsp
	popq	%rbp
	retq
.Ltmp3:
	.size __xray_FunctionExit, .Ltmp3-__xray_FunctionExit
	.cfi_endproc
