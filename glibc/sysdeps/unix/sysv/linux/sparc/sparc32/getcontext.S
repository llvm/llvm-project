/* Save current context.
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

#include "ucontext_i.h"

/*  int __getcontext (ucontext_t *ucp)

  Saves the machine context in UCP such that when it is activated,
  it appears as if __getcontext() returned again.

  This implementation is intended to be used for *synchronous* context
  switches only.  Therefore, it does not have to save anything
  other than the PRESERVED state.  */


ENTRY(__getcontext)
	save	%sp, -112, %sp
	st	%g0, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_PSR]

	/* In reality, we only use the GREG_PC value when setting
	   or swapping contexts.  But we fill in NPC for completeness.  */
	add	%i7, 8, %o0
	st	%o0, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_PC]
	add	%o0, 4, %o0
	st	%o0, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_NPC]

	rd	%y, %o1
	st	%o1, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_Y]

	st	%g1, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_G1]
	st	%g2, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_G2]
	st	%g3, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_G3]
	st	%g4, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_G4]
	st	%g5, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_G5]
	st	%g6, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_G6]
	st	%g7, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_G7]

	mov	SIG_BLOCK, %o0
	clr	%o1
	add	%i0, UC_SIGMASK, %o2
	mov	8, %o3
	mov	__NR_rt_sigprocmask, %g1
	ta	0x10

	/* Zero, success, return value.  */
	st	%g0, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_O0]
	st	%i1, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_O1]
	st	%i2, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_O2]
	st	%i3, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_O3]
	st	%i4, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_O4]
	st	%i5, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_O5]
	st	%i6, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_O6]
	st	%i7, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_O7]

	st	%g0, [%i0 + UC_MCONTEXT + MC_GWINS]

	/* Do not save FPU state, it is volatile across calls.  */
	stb	%g0, [%i0 + UC_MCONTEXT + MC_FPREGS + FPU_EN]

	st	%g0, [%i0 + UC_MCONTEXT + MC_XRS + XRS_ID]
	st	%g0, [%i0 + UC_MCONTEXT + MC_XRS + XRS_PTR]
	jmpl	%i7 + 8, %g0
	 restore %g0, %g0, %o0
END(__getcontext)

weak_alias (__getcontext, getcontext)
