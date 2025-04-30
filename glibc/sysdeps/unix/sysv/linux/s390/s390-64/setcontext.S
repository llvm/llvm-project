/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com).

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
#include <features.h>

#include "ucontext_i.h"

/*  __setcontext (const ucontext_t *ucp)

  Restores the machine context in UCP and thereby resumes execution
  in that context.

  This implementation is intended to be used for *synchronous* context
  switches only.  Therefore, it does not have to restore anything
  other than the PRESERVED state.  */

ENTRY(__setcontext)
	lgr	%r1,%r2

	/* sigprocmask (SIG_SETMASK, &sc->sc_mask, NULL).  */
	la      %r2,SIG_SETMASK
	la	%r3,SC_MASK(%r1)
	slgr	%r4,%r4
	lghi	%r5,_NSIG8
	svc	SYS_ify(rt_sigprocmask)

	/* Load fpu context.  */
	lfpc	SC_FPC(%r1)
	ld	%f0,SC_FPRS(%r1)
	ld      %f1,SC_FPRS+8(%r1)
	ld      %f2,SC_FPRS+16(%r1)
	ld      %f3,SC_FPRS+24(%r1)
	ld      %f4,SC_FPRS+32(%r1)
	ld      %f5,SC_FPRS+40(%r1)
	ld      %f6,SC_FPRS+48(%r1)
	ld      %f7,SC_FPRS+56(%r1)
	ld      %f8,SC_FPRS+64(%r1)
	ld      %f9,SC_FPRS+72(%r1)
	ld      %f10,SC_FPRS+80(%r1)
	ld      %f11,SC_FPRS+88(%r1)
	ld      %f12,SC_FPRS+96(%r1)
	ld      %f13,SC_FPRS+104(%r1)
	ld      %f14,SC_FPRS+112(%r1)
	ld      %f15,SC_FPRS+120(%r1)

	/* Don't touch %a0 and %a1, used for thread purposes.  */
	lam     %a2,%a15,SC_ACRS+8(%r1)

	/* Load general purpose registers.  */
	lmg	%r0,%r15,SC_GPRS(%r1)

	/* Return.  */
	br	%r14
END(__setcontext)

weak_alias (__setcontext, setcontext)
