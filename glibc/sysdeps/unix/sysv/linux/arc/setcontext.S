/* Set current context for ARC.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include "ucontext-macros.h"

/* int setcontext (const ucontext_t *ucp)
     - Restores the machine context in @ucp and resumes execution
       (doesn't return to caller).  */

ENTRY (__setcontext)

	mov  r9, r0	/* Stash @ucp across syscall.  */

	/* rt_sigprocmask (SIG_SETMASK, &ucp->uc_sigmask, NULL, _NSIG8).  */
	mov  r3, _NSIG8
	mov  r2, 0
	add  r1, r0, UCONTEXT_SIGMASK
	mov  r0, SIG_SETMASK
	mov  r8, __NR_rt_sigprocmask
	ARC_TRAP_INSN
	brhi r0, -1024, L (call_syscall_err)

	/* Restore scratch/arg regs for makecontext case.  */
	add r9, r9, UCONTEXT_MCONTEXT
	LDR (r0,    r9, 22)
	LDR (r1,    r9, 21)
	LDR (r2,    r9, 20)
	LDR (r3,    r9, 19)
	LDR (r4,    r9, 18)
	LDR (r5,    r9, 17)
	LDR (r6,    r9, 16)
	LDR (r7,    r9, 15)

	/* Restore callee saved registers.  */
	LDR (r13,   r9, 37)
	LDR (r14,   r9, 36)
	LDR (r15,   r9, 35)
	LDR (r16,   r9, 34)
	LDR (r17,   r9, 33)
	LDR (r18,   r9, 32)
	LDR (r19,   r9, 31)
	LDR (r20,   r9, 30)
	LDR (r21,   r9, 29)
	LDR (r22,   r9, 28)
	LDR (r23,   r9, 27)
	LDR (r24,   r9, 26)

	LDR (blink, r9,  7)
	LDR (fp,    r9,  8)
	LDR (gp,    r9,  9)
	LDR (sp,    r9, 23)

	j    [blink]

PSEUDO_END (__setcontext)
weak_alias (__setcontext, setcontext)


/* Helper for activating makecontext created context
     - r14 has @func, r15 has uc_link.  */

ENTRY (__startcontext)

	.cfi_label .Ldummy
	cfi_undefined (blink)

	/* Call user @func, loaded in r14 by setcontext.  */
	jl   [r14]

	/* If uc_link (r15) call setcontext with that.  */
	mov  r0, r15
	breq r0, 0, 1f

	bl   __setcontext
1:
	/* Exit with status 0.  */
	b    HIDDEN_JUMPTARGET(exit)
END (__startcontext)
