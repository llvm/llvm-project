/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>
#include <jmpbuf-offsets.h>
#include <jmp_buf-ssp.h>
#include <asm-syntax.h>
#include <stap-probe.h>

/* Don't restore shadow stack register if
   1. Shadow stack isn't enabled.  Or
   2. __longjmp is defined for __longjmp_cancel.
 */
#if !SHSTK_ENABLED || defined __longjmp
# undef SHADOW_STACK_POINTER_OFFSET
#endif

/* Jump to the position specified by ENV, causing the
   setjmp call there to return VAL, or 1 if VAL is 0.
   void __longjmp (__jmp_buf env, int val).  */
	.text
ENTRY(__longjmp)
	/* Restore registers.  */
	mov (JB_RSP*8)(%rdi),%R8_LP
	mov (JB_RBP*8)(%rdi),%R9_LP
	mov (JB_PC*8)(%rdi),%RDX_LP
#ifdef PTR_DEMANGLE
	PTR_DEMANGLE (%R8_LP)
	PTR_DEMANGLE (%R9_LP)
	PTR_DEMANGLE (%RDX_LP)
# ifdef __ILP32__
	/* We ignored the high bits of the %rbp value because only the low
	   bits are mangled.  But we cannot presume that %rbp is being used
	   as a pointer and truncate it, so recover the high bits.  */
	movl (JB_RBP*8 + 4)(%rdi), %eax
	shlq $32, %rax
	orq %rax, %r9
# endif
#endif
#ifdef SHADOW_STACK_POINTER_OFFSET
# if IS_IN (libc) && defined SHARED && defined FEATURE_1_OFFSET
	/* Check if Shadow Stack is enabled.  */
	testl $X86_FEATURE_1_SHSTK, %fs:FEATURE_1_OFFSET
	jz L(skip_ssp)
# else
	xorl %eax, %eax
# endif
	/* Check and adjust the Shadow-Stack-Pointer.  */
	/* Get the current ssp.  */
	rdsspq %rax
	/* And compare it with the saved ssp value.  */
	subq SHADOW_STACK_POINTER_OFFSET(%rdi), %rax
	je L(skip_ssp)
	/* Count the number of frames to adjust and adjust it
	   with incssp instruction.  The instruction can adjust
	   the ssp by [0..255] value only thus use a loop if
	   the number of frames is bigger than 255.  */
	negq %rax
	shrq $3, %rax
	/* NB: We saved Shadow-Stack-Pointer of setjmp.  Since we are
	       restoring Shadow-Stack-Pointer of setjmp's caller, we
	       need to unwind shadow stack by one more frame.  */
	addq $1, %rax

	movl $255, %ebx
L(loop):
	cmpq %rbx, %rax
	cmovb %rax, %rbx
	incsspq %rbx
	subq %rbx, %rax
	ja L(loop)

L(skip_ssp):
#endif
	LIBC_PROBE (longjmp, 3, LP_SIZE@%RDI_LP, -4@%esi, LP_SIZE@%RDX_LP)
	/* We add unwind information for the target here.  */
	cfi_def_cfa(%rdi, 0)
	cfi_register(%rsp,%r8)
	cfi_register(%rbp,%r9)
	cfi_register(%rip,%rdx)
	cfi_offset(%rbx,JB_RBX*8)
	cfi_offset(%r12,JB_R12*8)
	cfi_offset(%r13,JB_R13*8)
	cfi_offset(%r14,JB_R14*8)
	cfi_offset(%r15,JB_R15*8)
	movq (JB_RBX*8)(%rdi),%rbx
	movq (JB_R12*8)(%rdi),%r12
	movq (JB_R13*8)(%rdi),%r13
	movq (JB_R14*8)(%rdi),%r14
	movq (JB_R15*8)(%rdi),%r15
	/* Set return value for setjmp.  */
	mov %esi, %eax
	mov %R8_LP,%RSP_LP
	movq %r9,%rbp
	LIBC_PROBE (longjmp_target, 3,
		    LP_SIZE@%RDI_LP, -4@%eax, LP_SIZE@%RDX_LP)
	jmpq *%rdx
END (__longjmp)
