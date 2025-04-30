/* Save current context for ARC.
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

/* int getcontext (ucontext_t *ucp)
   Save machine context in @ucp and return 0 on success, -1 on error
    - saves callee saved registers only
    - layout mandated by uncontext_t:uc_mcontext (hence different from setjmp).  */

ENTRY (__getcontext)

	/* Callee saved registers.  */
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

	/* Save 0 in r0 placeholder to return 0 when this @ucp activated.  */
	mov r9, 0
	STR (r9,    r2, 22)

	/* rt_sigprocmask (SIG_BLOCK, NULL, &ucp->uc_sigmask, _NSIG8).  */
	mov r3, _NSIG8
	add r2, r0, UCONTEXT_SIGMASK
	mov r1, 0
	mov r0, SIG_BLOCK
	mov r8, __NR_rt_sigprocmask
	ARC_TRAP_INSN
	brhi    r0, -1024, L (call_syscall_err)
	j.d	[blink]
	mov r0, 0	/* Success, error handled in .Lcall_syscall_err.  */

PSEUDO_END (__getcontext)
weak_alias (__getcontext, getcontext)
