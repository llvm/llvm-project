/* Install given context.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David S. Miller <davem@davemloft.net>, 2008.

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
#include <sys/trap.h>

#include "ucontext_i.h"


/*  int __setcontext (const ucontext_t *ucp)

  Restores the machine context in UCP and thereby resumes execution
  in that context.

  This implementation is intended to be used for *synchronous* context
  switches only.  Therefore, it does not have to restore anything
  other than the PRESERVED state.  */

ENTRY(__setcontext)
	save	%sp, -112, %sp

	mov	SIG_SETMASK, %o0
	add	%i0, UC_SIGMASK, %o1
	clr	%o2
	mov	8, %o3
	mov	__NR_rt_sigprocmask, %g1
	ta	0x10

	/* This is a bit on the expensive side, and we could optimize
	   the unwind similar to how the 32-bit sparc longjmp code
	   does if performance of this routine really matters.  */
	ta	ST_FLUSH_WINDOWS

	ldub	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_EN], %g1
	cmp	%g1, 0
	be	1f
	 nop
	ld	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_FSR], %fsr
	ldd	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_D0], %f0
	ldd	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_D2], %f2
	ldd	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_D4], %f4
	ldd	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_D6], %f6
	ldd	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_D8], %f8
	ldd	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_D10], %f10
	ldd	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_D12], %f12
	ldd	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_D14], %f14
	ldd	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_D16], %f16
	ldd	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_D18], %f18
	ldd	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_D20], %f20
	ldd	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_D22], %f22
	ldd	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_D24], %f24
	ldd	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_D26], %f26
	ldd	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_D28], %f28
	ldd	[%i0 + UC_MCONTEXT + MC_FPREGS + FPU_D30], %f30
1:
	ld	[%i0 + UC_MCONTEXT + MC_GREGS + GREG_Y], %g1
	wr	%g1, 0x0, %y

	/* We specifically do not restore %g1 since we need it here as
	   a temporary.  */
	ld	[%i0 + UC_MCONTEXT + MC_GREGS + GREG_G2], %g2
	ld	[%i0 + UC_MCONTEXT + MC_GREGS + GREG_G3], %g3
	ld	[%i0 + UC_MCONTEXT + MC_GREGS + GREG_G4], %g4
	ld	[%i0 + UC_MCONTEXT + MC_GREGS + GREG_G5], %g5
	ld	[%i0 + UC_MCONTEXT + MC_GREGS + GREG_G6], %g6
	ld	[%i0 + UC_MCONTEXT + MC_GREGS + GREG_G7], %g7
	ld	[%i0 + UC_MCONTEXT + MC_GREGS + GREG_O1], %i1
	ld	[%i0 + UC_MCONTEXT + MC_GREGS + GREG_O2], %i2
	ld	[%i0 + UC_MCONTEXT + MC_GREGS + GREG_O3], %i3
	ld	[%i0 + UC_MCONTEXT + MC_GREGS + GREG_O4], %i4
	ld	[%i0 + UC_MCONTEXT + MC_GREGS + GREG_O5], %i5
	ld	[%i0 + UC_MCONTEXT + MC_GREGS + GREG_O6], %i6
	restore
	ld	[%o0 + UC_MCONTEXT + MC_GREGS + GREG_O7], %o7
	ld	[%o0 + UC_MCONTEXT + MC_GREGS + GREG_PC], %g1
	jmpl	%g1, %g0
	 ld	[%o0 + UC_MCONTEXT + MC_GREGS + GREG_O0], %o0
END(__setcontext)

weak_alias (__setcontext, setcontext)

/* We add an NOP here to separate between __setcontext/__startcontext. The
   wanted behavior that happens is: when unwinding from a function called
   inside a makecontext() context, FDE lookup will use '&__startcontext - 1',
   then returns NULL for no FDE found, and immediately ends the unwind, in
   a normal fashion.

   If this NOP word does not exist, FDE lookup just repeatedly finds
   __setcontext's FDE in an infinite loop, due to the convention of using
   'address - 1' for FDE lookup. Modifiying/deleting the below
   __startcontext's FDE has no help on this.  */

	nop

/* This is the helper code which gets called if a function which is
   registered with 'makecontext' returns.  In this case we have to
   install the context listed in the uc_link element of the context
   'makecontext' manipulated at the time of the 'makecontext' call.
   If the pointer is NULL the process must terminate.  */

ENTRY(__start_context)
	ld	[%sp + (16 * 4)], %g1
	cmp	%g1, 0
	be,a	1f
	 clr	%o0
	call	__setcontext
	 mov	%g1, %o0
	/* If this returns (which can happen if the syscall fails) we'll
	   exit the program with the return error value (-1).  */
1:	call	HIDDEN_JUMPTARGET(exit)
	 nop
	/* The 'exit' call should never return.  In case it does cause
	   the process to terminate.  */
	unimp
END(__start_context)
