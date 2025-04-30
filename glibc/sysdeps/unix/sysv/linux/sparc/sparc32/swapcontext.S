/* Save current context and install the given one.
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


/* int __swapcontext (ucontext_t *oucp, const ucontext_t *ucp);

  Saves the machine context in oucp such that when it is activated,
  it appears as if __swapcontext() returned again, restores the
  machine context in ucp and thereby resumes execution in that
  context.

  This implementation is intended to be used for *synchronous* context
  switches only.  Therefore, it does not have to save anything
  other than the PRESERVED state.  */

ENTRY(__swapcontext)
	save	%sp, -112, %sp
	ta	ST_FLUSH_WINDOWS
	st	%g0, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_PSR]
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
	st	%g0, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_O0]
	st	%i1, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_O1]
	st	%i2, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_O2]
	st	%i3, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_O3]
	st	%i4, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_O4]
	st	%i5, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_O5]
	st	%i6, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_O6]
	st	%i7, [%i0 + UC_MCONTEXT + MC_GREGS + GREG_O7]
	st	%g0, [%i0 + UC_MCONTEXT + MC_GWINS]
	stb	%g0, [%i0 + UC_MCONTEXT + MC_FPREGS + FPU_EN]
	st	%g0, [%i0 + UC_MCONTEXT + MC_XRS + XRS_ID]
	st	%g0, [%i0 + UC_MCONTEXT + MC_XRS + XRS_PTR]

	mov	SIG_SETMASK, %o0
	add	%i1, UC_SIGMASK, %o1
	add	%i0, UC_SIGMASK, %o2
	mov	8, %o3
	mov	__NR_rt_sigprocmask, %g1
	ta	0x10

	mov	%i1, %i0
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
END(__swapcontext)

weak_alias (__swapcontext, swapcontext)
