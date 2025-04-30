/* longjmp for i386.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

	.text
ENTRY (__longjmp)
#ifdef PTR_DEMANGLE
	movl 4(%esp), %eax	/* User's jmp_buf in %eax.  */

# ifdef SHADOW_STACK_POINTER_OFFSET
#  if IS_IN (libc) && defined SHARED && defined FEATURE_1_OFFSET
	/* Check if Shadow Stack is enabled.  */
	testl $X86_FEATURE_1_SHSTK, %gs:FEATURE_1_OFFSET
	jz L(skip_ssp)
#  else
	xorl %edx, %edx
#  endif
	/* Check and adjust the Shadow-Stack-Pointer.  */
	rdsspd %edx
	/* And compare it with the saved ssp value.  */
	subl SHADOW_STACK_POINTER_OFFSET(%eax), %edx
	je L(skip_ssp)
	/* Count the number of frames to adjust and adjust it
	   with incssp instruction.  The instruction can adjust
	   the ssp by [0..255] value only thus use a loop if
	   the number of frames is bigger than 255.  */
	negl %edx
	shrl $2, %edx
	/* NB: We saved Shadow-Stack-Pointer of setjmp.  Since we are
	       restoring Shadow-Stack-Pointer of setjmp's caller, we
	       need to unwind shadow stack by one more frame.  */
	addl $1, %edx
	movl	$255, %ebx
L(loop):
	cmpl	%ebx, %edx
	cmovb	%edx, %ebx
	incsspd	%ebx
	subl	%ebx, %edx
	ja	L(loop)
L(skip_ssp):
# endif
	/* Save the return address now.  */
	movl (JB_PC*4)(%eax), %edx
	/* Get the stack pointer.  */
	movl (JB_SP*4)(%eax), %ecx
	PTR_DEMANGLE (%edx)
	PTR_DEMANGLE (%ecx)
	LIBC_PROBE (longjmp, 3, 4@%eax, -4@8(%esp), 4@%edx)
	cfi_def_cfa(%eax, 0)
	cfi_register(%eip, %edx)
	cfi_register(%esp, %ecx)
	cfi_offset(%ebx, JB_BX*4)
	cfi_offset(%esi, JB_SI*4)
	cfi_offset(%edi, JB_DI*4)
	cfi_offset(%ebp, JB_BP*4)
	/* Restore registers.  */
	movl (JB_BX*4)(%eax), %ebx
	movl (JB_SI*4)(%eax), %esi
	movl (JB_DI*4)(%eax), %edi
	movl (JB_BP*4)(%eax), %ebp
	cfi_restore(%ebx)
	cfi_restore(%esi)
	cfi_restore(%edi)
	cfi_restore(%ebp)

	LIBC_PROBE (longjmp_target, 3, 4@%eax, -4@8(%esp), 4@%edx)
	movl 8(%esp), %eax	/* Second argument is return value.  */
	movl %ecx, %esp
#else
	movl 4(%esp), %ecx	/* User's jmp_buf in %ecx.  */
	movl 8(%esp), %eax	/* Second argument is return value.  */
# ifdef SHADOW_STACK_POINTER_OFFSET
#  if IS_IN (libc) && defined SHARED
	/* Check if Shadow Stack is enabled.  */
	testl $X86_FEATURE_1_SHSTK, %gs:FEATURE_1_OFFSET
	jz L(skip_ssp)
#  endif
	/* Check and adjust the Shadow-Stack-Pointer.  */
	xorl %edx, %edx
	/* Get the current ssp.  */
	rdsspd	%edx
	/* And compare it with the saved ssp value.  */
	subl SHADOW_STACK_POINTER_OFFSET(%ecx), %edx
	je L(skip_ssp)
	/* Count the number of frames to adjust and adjust it
	   with incssp instruction.  The instruction can adjust
	   the ssp by [0..255] value only thus use a loop if
	   the number of frames is bigger than 255.  */
	negl %edx
	shrl $2, %edx
	/* NB: We saved Shadow-Stack-Pointer of setjmp.  Since we are
	       restoring Shadow-Stack-Pointer of setjmp's caller, we
	       need to unwind shadow stack by one more frame.  */
	addl $1, %edx
	movl	$255, %ebx
L(loop):
	cmpl	%ebx, %edx
	cmovb	%edx, %ebx
	incsspd	%ebx
	subl	%ebx, %edx
	ja	L(loop)
L(skip_ssp):
# endif
	/* Save the return address now.  */
	movl (JB_PC*4)(%ecx), %edx
	LIBC_PROBE (longjmp, 3, 4@%ecx, -4@%eax, 4@%edx)
	/* Restore registers.  */
	movl (JB_BX*4)(%ecx), %ebx
	movl (JB_SI*4)(%ecx), %esi
	movl (JB_DI*4)(%ecx), %edi
	movl (JB_BP*4)(%ecx), %ebp
	movl (JB_SP*4)(%ecx), %esp
	LIBC_PROBE (longjmp_target, 3, 4@%ecx, -4@%ecx, 4@%edx)
#endif
	/* Jump to saved PC.  */
	jmp *%edx
END (__longjmp)
