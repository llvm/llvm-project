/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

/*  __swapcontext (ucontext_t *oucp, const ucontext_t *ucp)

  Saves the machine context in oucp such that when it is activated,
  it appears as if __swapcontext() returned again, restores the
  machine context in ucp and thereby resumes execution in that
  context.

  This implementation is intended to be used for *synchronous* context
  switches only.  Therefore, it does not have to save anything
  other than the PRESERVED state.  */

ENTRY(__swapcontext)
	/* While not part of the ABI a system call never clobbers r0
	   or r1.  So keeping the values here while calling
	   rt_sigprocmask is ok.  */
	lr	%r1,%r2
	lr      %r0,%r3

	/* Store fpu context.  */
	stfpc   SC_FPC(%r1)
	std     %f0,SC_FPRS(%r1)
	std     %f1,SC_FPRS+8(%r1)
	std     %f2,SC_FPRS+16(%r1)
	std     %f3,SC_FPRS+24(%r1)
	std     %f4,SC_FPRS+32(%r1)
	std     %f5,SC_FPRS+40(%r1)
	std     %f6,SC_FPRS+48(%r1)
	std     %f7,SC_FPRS+56(%r1)
	std     %f8,SC_FPRS+64(%r1)
	std     %f9,SC_FPRS+72(%r1)
	std     %f10,SC_FPRS+80(%r1)
	std     %f11,SC_FPRS+88(%r1)
	std     %f12,SC_FPRS+96(%r1)
	std     %f13,SC_FPRS+104(%r1)
	std     %f14,SC_FPRS+112(%r1)
	std     %f15,SC_FPRS+120(%r1)

	/* Set __swapcontext return value to 0.  */
	slr     %r2,%r2

	/* Store access registers.  */
	stam    %a0,%a15,SC_ACRS(%r1)

	/* Store general purpose registers.  */
	stm     %r0,%r15,SC_GPRS(%r1)

	/* rt_sigprocmask (SIG_SETMASK, &ucp->uc_sigmask, &oucp->uc_sigmask,
	   sigsetsize).  */
	la      %r2,SIG_SETMASK
	lr	%r5,%r0
	la	%r3,SC_MASK(%r5)
	la	%r4,SC_MASK(%r1)
	lhi	%r5,_NSIG8
	svc	SYS_ify(rt_sigprocmask)

	/* Load fpu context.  */
	lr	%r5,%r0
	lfpc	SC_FPC(%r5)
	ld	%f0,SC_FPRS(%r5)
	ld      %f1,SC_FPRS+8(%r5)
	ld      %f2,SC_FPRS+16(%r5)
	ld      %f3,SC_FPRS+24(%r5)
	ld      %f4,SC_FPRS+32(%r5)
	ld      %f5,SC_FPRS+40(%r5)
	ld      %f6,SC_FPRS+48(%r5)
	ld      %f7,SC_FPRS+56(%r5)
	ld      %f8,SC_FPRS+64(%r5)
	ld      %f9,SC_FPRS+72(%r5)
	ld      %f10,SC_FPRS+80(%r5)
	ld      %f11,SC_FPRS+88(%r5)
	ld      %f12,SC_FPRS+96(%r5)
	ld      %f13,SC_FPRS+104(%r5)
	ld      %f14,SC_FPRS+112(%r5)
	ld      %f15,SC_FPRS+120(%r5)

	/* Don't touch %a0, used for thread purposes.  */
	lam	%a1,%a15,SC_ACRS+4(%r5)

	/* Load general purpose registers.  */
	lm	%r0,%r15,SC_GPRS(%r5)

	/* Return.  */
	br	%r14
END(__swapcontext)
weak_alias (__swapcontext, swapcontext)
