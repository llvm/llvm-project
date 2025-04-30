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
#include <tcb-offsets.h>
#include <asm-syntax.h>

#include <signal-defines.h>
/* #include <signal.h> */
#define SS_ONSTACK 1

	.text
ENTRY (__longjmp)
	movl 4(%esp), %eax	/* User's jmp_buf in %eax.  */

	/* Save the return address now.  */
	movl (JB_PC*4)(%eax), %edx
	/* Get the stack pointer.  */
	movl (JB_SP*4)(%eax), %ecx
	cfi_undefined(%ecx)
#ifdef PTR_DEMANGLE
	PTR_DEMANGLE (%edx)
	PTR_DEMANGLE (%ecx)
#endif

	movw	%ds, %si
	movw	%gs, %di
	cmpw	%si, %di
	jz	.Lok		/* TLS not initialized yet */

	movl	%gs:SIGSTATE_OFFSET, %edi
	testl	%edi, %edi
	jz	.Lok		/* sigstate not initialized yet */

	testl	$SS_ONSTACK, (HURD_SIGSTATE__SIGALTSTACK__OFFSET + SIGALTSTACK__SS_FLAGS__OFFSET)(%edi)
	jz	.Lok

	/* We were on the alternate stack.  */

	cmpl	(HURD_SIGSTATE__SIGALTSTACK__OFFSET + SIGALTSTACK__SS_SP__OFFSET)(%edi), %ecx
	jb	.Loks		/* We jump below the alternate stack, switch.  */

	movl	(HURD_SIGSTATE__SIGALTSTACK__OFFSET + SIGALTSTACK__SS_SP__OFFSET)(%edi), %ebx
	addl	(HURD_SIGSTATE__SIGALTSTACK__OFFSET + SIGALTSTACK__SS_SIZE__OFFSET)(%edi), %ebx
	cmpl	%ebx, %ecx
	jb	.Lok		/* We jump inside the alternate stack, do not switch.  */

	/* We jump above the alternate stack, switch.  */

.Loks:	/* We jump out of the alternate stack, clear SS_ONSTACK flag.  */
	andl	$~(SS_ONSTACK), (HURD_SIGSTATE__SIGALTSTACK__OFFSET + SIGALTSTACK__SS_FLAGS__OFFSET)(%edi)

.Lok:	/* We add unwind information for the target here.  */
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

	movl	8(%esp), %eax	/* Second argument is return value.  */
	movl	%ecx, %esp

	/* Jump to saved PC.  */
	jmp *%edx
END (__longjmp)
