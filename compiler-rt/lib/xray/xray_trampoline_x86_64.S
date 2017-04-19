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

#include "../builtins/assembly.h"

.macro SAVE_REGISTERS
	subq $184, %rsp
	movupd	%xmm0, 168(%rsp)
	movupd	%xmm1, 152(%rsp)
	movupd	%xmm2, 136(%rsp)
	movupd	%xmm3, 120(%rsp)
	movupd	%xmm4, 104(%rsp)
	movupd	%xmm5, 88(%rsp)
	movupd	%xmm6, 72(%rsp)
	movupd	%xmm7, 56(%rsp)
	movq	%rdi, 48(%rsp)
	movq	%rax, 40(%rsp)
	movq	%rdx, 32(%rsp)
	movq	%rsi, 24(%rsp)
	movq	%rcx, 16(%rsp)
	movq	%r8, 8(%rsp)
	movq	%r9, 0(%rsp)
.endm

.macro RESTORE_REGISTERS
	movupd	168(%rsp), %xmm0
	movupd	152(%rsp), %xmm1
	movupd	136(%rsp), %xmm2
	movupd	120(%rsp), %xmm3
	movupd	104(%rsp), %xmm4
	movupd	88(%rsp), %xmm5
	movupd	72(%rsp) , %xmm6
	movupd	56(%rsp) , %xmm7
	movq	48(%rsp), %rdi
	movq	40(%rsp), %rax
	movq	32(%rsp), %rdx
	movq	24(%rsp), %rsi
	movq	16(%rsp), %rcx
	movq	8(%rsp), %r8
	movq	0(%rsp), %r9
	addq	$184, %rsp
.endm

	.text
	.file "xray_trampoline_x86.S"

//===----------------------------------------------------------------------===//

	.globl __xray_FunctionEntry
	.align 16, 0x90
	.type __xray_FunctionEntry,@function

__xray_FunctionEntry:
	.cfi_startproc
	pushq %rbp
	.cfi_def_cfa_offset 16
	SAVE_REGISTERS
	.cfi_def_cfa_offset 200

	// This load has to be atomic, it's concurrent with __xray_patch().
	// On x86/amd64, a simple (type-aligned) MOV instruction is enough.
	movq	_ZN6__xray19XRayPatchedFunctionE(%rip), %rax
	testq	%rax, %rax
	je	.Ltmp0

	// The patched function prolog puts its xray_instr_map index into %r10d.
	movl	%r10d, %edi
	xor	%esi,%esi
	callq	*%rax
.Ltmp0:
	RESTORE_REGISTERS
	popq	%rbp
	retq
.Ltmp1:
	.size __xray_FunctionEntry, .Ltmp1-__xray_FunctionEntry
	.cfi_endproc

//===----------------------------------------------------------------------===//

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
	subq	$48, %rsp
	.cfi_def_cfa_offset 64
	movupd	%xmm0, 32(%rsp)
	movupd	%xmm1, 16(%rsp)
	movq	%rax, 8(%rsp)
	movq	%rdx, 0(%rsp)
	movq	_ZN6__xray19XRayPatchedFunctionE(%rip), %rax
	testq %rax,%rax
	je	.Ltmp2

	movl	%r10d, %edi
	movl	$1, %esi
	callq	*%rax
.Ltmp2:
	// Restore the important registers.
	movupd	32(%rsp), %xmm0
	movupd	16(%rsp), %xmm1
	movq	8(%rsp), %rax
	movq	0(%rsp), %rdx
	addq	$48, %rsp
	popq	%rbp
	retq
.Ltmp3:
	.size __xray_FunctionExit, .Ltmp3-__xray_FunctionExit
	.cfi_endproc

//===----------------------------------------------------------------------===//

	.global __xray_FunctionTailExit
	.align 16, 0x90
	.type __xray_FunctionTailExit,@function
__xray_FunctionTailExit:
	.cfi_startproc
	// Save the important registers as in the entry trampoline, but indicate that
	// this is an exit. In the future, we will introduce a new entry type that
	// differentiates between a normal exit and a tail exit, but we'd have to do
	// this and increment the version number for the header.
	pushq %rbp
	.cfi_def_cfa_offset 16
	SAVE_REGISTERS
	.cfi_def_cfa_offset 200

	movq	_ZN6__xray19XRayPatchedFunctionE(%rip), %rax
	testq %rax,%rax
	je	.Ltmp4

	movl	%r10d, %edi
	movl	$1, %esi
	callq	*%rax

.Ltmp4:
	RESTORE_REGISTERS
	popq	%rbp
	retq
.Ltmp5:
	.size __xray_FunctionTailExit, .Ltmp5-__xray_FunctionTailExit
	.cfi_endproc

//===----------------------------------------------------------------------===//

	.globl __xray_ArgLoggerEntry
	.align 16, 0x90
	.type __xray_ArgLoggerEntry,@function
__xray_ArgLoggerEntry:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	SAVE_REGISTERS
	.cfi_def_cfa_offset 200

	// Again, these function pointer loads must be atomic; MOV is fine.
	movq	_ZN6__xray13XRayArgLoggerE(%rip), %rax
	testq	%rax, %rax
	jne	.Larg1entryLog

	// If [arg1 logging handler] not set, defer to no-arg logging.
	movq	_ZN6__xray19XRayPatchedFunctionE(%rip), %rax
	testq	%rax, %rax
	je	.Larg1entryFail

.Larg1entryLog:
	movq	%rdi, %rdx	// first argument will become the third
	xorq	%rsi, %rsi	// XRayEntryType::ENTRY into the second
	movl	%r10d, %edi	// 32-bit function ID becomes the first
	callq	*%rax

.Larg1entryFail:
	RESTORE_REGISTERS
	popq	%rbp
	retq

.Larg1entryEnd:
	.size __xray_ArgLoggerEntry, .Larg1entryEnd-__xray_ArgLoggerEntry
	.cfi_endproc

NO_EXEC_STACK_DIRECTIVE
