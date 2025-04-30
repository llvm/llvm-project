/* setjmp for i386.
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

#define PARMS	4		/* no space for saved regs */
#define JMPBUF	PARMS
#define SIGMSK	JMPBUF+4

/* Don't save shadow stack register if shadow stack isn't enabled.  */
#if !SHSTK_ENABLED
# undef SHADOW_STACK_POINTER_OFFSET
#endif

ENTRY (__sigsetjmp)

	movl JMPBUF(%esp), %eax

	/* Save registers.  */
	movl %ebx, (JB_BX*4)(%eax)
	movl %esi, (JB_SI*4)(%eax)
	movl %edi, (JB_DI*4)(%eax)
	leal JMPBUF(%esp), %ecx	/* Save SP as it will be after we return.  */
#ifdef PTR_MANGLE
	PTR_MANGLE (%ecx)
#endif
	movl %ecx, (JB_SP*4)(%eax)
	movl 0(%esp), %ecx	/* Save PC we are returning to now.  */
	LIBC_PROBE (setjmp, 3, 4@%eax, -4@SIGMSK(%esp), 4@%ecx)
#ifdef PTR_MANGLE
	PTR_MANGLE (%ecx)
#endif
	movl %ecx, (JB_PC*4)(%eax)
	movl %ebp, (JB_BP*4)(%eax) /* Save caller's frame pointer.  */

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
	movl %ecx, SHADOW_STACK_POINTER_OFFSET(%eax)
# if IS_IN (libc) && defined SHARED && defined FEATURE_1_OFFSET
L(skip_ssp):
# endif
#endif
#if IS_IN (rtld)
	/* In ld.so we never save the signal mask.  */
	xorl %eax, %eax
	ret
#else
	/* Make a tail call to __sigjmp_save; it takes the same args.  */
	jmp __sigjmp_save
#endif
END (__sigsetjmp)
hidden_def (__sigsetjmp)
