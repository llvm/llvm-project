/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Mosberger-Tang <davidm@hpl.hp.com>.

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

/*  __getcontext (ucontext_t *ucp)

  Saves the machine context in UCP such that when it is activated,
  it appears as if __getcontext() returned again.  The only difference
  is that on a first return, r9 contains 1 and on a subsequent
  return, it contains 0.

  This implementation in intended to be used for *synchronous* context
  switches only.  Therefore, it does not have to save anything
  other than the PRESERVED state.  */

ENTRY(__getcontext)
	.prologue
	.body
	alloc r11 = ar.pfs, 1, 0, 4, 0

	// sigprocmask (SIG_BLOCK, NULL, &sc->sc_mask):

	mov r3 = SC_MASK
	mov out0 = SIG_BLOCK

	flushrs					// save dirty partition on rbs
	mov out1 = 0
	add out2 = r3, in0

	mov out3 = 8	// sizeof kernel sigset_t
	DO_CALL(__NR_rt_sigprocmask)

	mov.m rFPSR = ar.fpsr
	mov.m rRSC = ar.rsc
	add r2 = SC_GR+1*8, r32
	;;
	mov.m rBSP = ar.bsp
	.prologue
	.save ar.unat, rUNAT
	mov.m rUNAT = ar.unat
	.body
	add r3 = SC_GR+4*8, r32
	;;

.mem.offset 0,0; st8.spill [r2] = r1, (5*8 - 1*8)
.mem.offset 8,0; st8.spill [r3] = r4, 16
	mov rPFS = r11
	;;
.mem.offset 0,0; st8.spill [r2] = r5, 16
.mem.offset 8,0; st8.spill [r3] = r6, 48
	and rTMP = ~0x3, rRSC
	;;
.mem.offset 0,0; st8.spill [r2] = r7, (SC_FR+2*16-(SC_GR+7*8))
.mem.offset 8,0; st8.spill [r3] = sp, (SC_FR+3*16-(SC_GR+12*8))
	;;
	mov.m ar.rsc = rTMP 		// put RSE into enforced lazy mode
	mov.m rNAT = ar.unat
	mov.i rLC = ar.lc
	;;
	mov.m rRNAT = ar.rnat
	mov.m ar.rsc = rRSC 		// restore RSE mode
	mov rPR = pr

	/*
	 * Rotate NaT bits by rPOS positions to the right:
	 */
	stf.spill [r2] = f2, 32
	stf.spill [r3] = f3, 32
	add rPOS = SC_GR, r32			// rPOS <- &sc_gr[0]
	;;
	stf.spill [r2] = f4, (16*16-4*16)
	stf.spill [r3] = f5, (17*16-5*16)
	extr.u rPOS = rPOS, 3, 6		// get NaT bit number for r0
	;;
	stf.spill [r2] = f16, 32
	stf.spill [r3] = f17, 32
	sub rCPOS = 64, rPOS
	;;
	stf.spill [r2] = f18, 32
	stf.spill [r3] = f19, 32
	shr.u rTMP = rNAT, rPOS
	;;
	stf.spill [r2] = f20, 32
	stf.spill [r3] = f21, 32
	shl rNAT = rNAT, rCPOS
	;;
	stf.spill [r2] = f22, 32
	stf.spill [r3] = f23, 32
	or rNAT = rNAT, rTMP
	;;
	stf.spill [r2] = f24, 32
	stf.spill [r3] = f25, 32
	mov r8 = 0
	;;
	stf.spill [r2] = f26, 32
	stf.spill [r3] = f27, 32
	mov r9 = 1
	;;
	stf.spill [r2] = f28, 32
	stf.spill [r3] = f29, 32
	mov rB0 = b0
	;;
	stf.spill [r2] = f30, 32
	stf.spill [r3] = f31, 32
	mov rB1 = b1
	;;
	mov ar.unat = rUNAT	// we're done spilling integer regs; restore caller's UNaT
	add r2 = SC_NAT, r32
	add r3 = SC_BSP, r32
	;;
	st8 [r2] = rNAT, (SC_RNAT-SC_NAT)
	st8 [r3] = rBSP, (SC_UNAT-SC_BSP)
	mov rB2 = b2
	;;
	st8 [r2] = rRNAT, (SC_FPSR-SC_RNAT)
	st8 [r3] = rUNAT, (SC_PFS-SC_UNAT)
	mov rB3 = b3
	;;
	st8 [r2] = rFPSR, (SC_LC-SC_FPSR)
	st8 [r3] = rPFS, (SC_PR-SC_PFS)
	mov rB4 = b4
	;;
	st8 [r2] = rLC, (SC_BR+0*8-SC_LC)
	st8 [r3] = rPR, (SC_BR+1*8-SC_PR)
	mov rB5 = b5
	;;
	st8 [r2] = rB0, 16
	st8 [r3] = rB1, 16
	;;
	st8 [r2] = rB2, 16
	st8 [r3] = rB3, 16
	;;
	st8 [r2] = rB4
	st8 [r3] = rB5
	ret
END(__getcontext)

weak_alias (__getcontext, getcontext)
