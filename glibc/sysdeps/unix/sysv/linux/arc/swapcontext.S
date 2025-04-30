/* Save and set current context for ARC.
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

/* int swapcontext (ucontext_t *oucp, const ucontext_t *ucp).  */

ENTRY (__swapcontext)

	/* Save context into @oucp pointed to by r0.  */

	add r2, r0, UCONTEXT_MCONTEXT
	STR (r13,   r2, 37)
	STR (r14,   r2, 36)
	STR (r15,   r2, 35)
	STR (r16,   r2, 34)
	STR (r17,   r2, 33)
	STR (r18,   r2, 32)
	STR (r19,   r2, 31)
	STR (r20,   r2, 30)
	STR (r21,   r2, 29)
	STR (r22,   r2, 28)
	STR (r23,   r2, 27)
	STR (r24,   r2, 26)

	STR (blink, r2,  7)
	STR (fp,    r2,  8)
	STR (gp,    r2,  9)
	STR (sp,    r2, 23)

	/* Save 0 in r0 placeholder to return 0 when @oucp activated.  */
	mov r9, 0
	STR (r9,    r2, 22)

	/* Load context from @ucp.  */

	mov r9, r1	/* Safekeep @ucp across syscall.  */

	/* rt_sigprocmask (SIG_SETMASK, &ucp->uc_sigmask, &oucp->uc_sigmask, _NSIG8) */
	mov r3, _NSIG8
	add r2, r0, UCONTEXT_SIGMASK
	add r1, r1, UCONTEXT_SIGMASK
	mov r0, SIG_SETMASK
	mov r8, __NR_rt_sigprocmask
	ARC_TRAP_INSN
	brhi r0, -1024, L (call_syscall_err)

	add r9, r9, UCONTEXT_MCONTEXT
	LDR (r0,    r9, 22)
	LDR (r1,    r9, 21)
	LDR (r2,    r9, 20)
	LDR (r3,    r9, 19)
	LDR (r4,    r9, 18)
	LDR (r5,    r9, 17)
	LDR (r6,    r9, 16)
	LDR (r7,    r9, 15)

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

PSEUDO_END (__swapcontext)
weak_alias (__swapcontext, swapcontext)
