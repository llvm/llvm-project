/* BSD `_setjmp' entry point to `sigsetjmp (..., 0)'.  i386 version.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

/* This just does a tail-call to `__sigsetjmp (ARG, 0)'.
   We cannot do it in C because it must be a tail-call, so frame-unwinding
   in setjmp doesn't clobber the state restored by longjmp.  */

#include <sysdep.h>
#include <jmpbuf-offsets.h>
#include <jmp_buf-ssp.h>
#include <stap-probe.h>

#define PARMS	4		/* no space for saved regs */
#define JMPBUF	PARMS
#define SIGMSK	JMPBUF+4

/* Don't save shadow stack register if shadow stack isn't enabled.  */
#if !SHSTK_ENABLED
# undef SHADOW_STACK_POINTER_OFFSET
#endif

ENTRY (_setjmp)

	xorl %eax, %eax
	movl JMPBUF(%esp), %edx

	/* Save registers.  */
	movl %ebx, (JB_BX*4)(%edx)
	movl %esi, (JB_SI*4)(%edx)
	movl %edi, (JB_DI*4)(%edx)
	leal JMPBUF(%esp), %ecx	/* Save SP as it will be after we return.  */
#ifdef PTR_MANGLE
	PTR_MANGLE (%ecx)
#endif
	movl %ecx, (JB_SP*4)(%edx)
	movl 0(%esp), %ecx	/* Save PC we are returning to now.  */
	LIBC_PROBE (setjmp, 3, 4@%edx, -4@$0, 4@%ecx)
#ifdef PTR_MANGLE
	PTR_MANGLE (%ecx)
#endif
	movl %ecx, (JB_PC*4)(%edx)
	movl %ebp, (JB_BP*4)(%edx) /* Save caller's frame pointer.  */

	movl %eax, JB_SIZE(%edx) /* No signal mask set.  */
#ifdef SHADOW_STACK_POINTER_OFFSET
# if IS_IN (libc) && defined SHARED && defined FEATURE_1_OFFSET
	/* Check if Shadow Stack is enabled.  */
	testl $X86_FEATURE_1_SHSTK, %gs:FEATURE_1_OFFSET
	jz L(skip_ssp)
# else
	xorl %ecx, %ecx
# endif
	/* Get the current Shadow-Stack-Pointer and save it.  */
	rdsspd %ecx
	movl %ecx, SHADOW_STACK_POINTER_OFFSET(%edx)
# if IS_IN (libc) && defined SHARED && defined FEATURE_1_OFFSET
L(skip_ssp):
# endif
#endif
	ret
END (_setjmp)
libc_hidden_def (_setjmp)
