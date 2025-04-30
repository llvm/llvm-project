/* Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
#include <rtld-global-offsets.h>

#include "ucontext_i.h"

	.syntax unified
	.text

/* int getcontext (ucontext_t *ucp) */

ENTRY(__getcontext)
	/* No need to save r0-r3, d0-d7, or d16-d31.  */
	add	r1, r0, #MCONTEXT_ARM_R4
	stmia   r1, {r4-r11}

	/* Save R13 separately as Thumb can't STM it.  */
	str     r13, [r0, #MCONTEXT_ARM_SP]
	str     r14, [r0, #MCONTEXT_ARM_LR]
	/* Return to LR */
	str     r14, [r0, #MCONTEXT_ARM_PC]
	/* Return zero */
	mov     r2, #0
	str     r2, [r0, #MCONTEXT_ARM_R0]

	/* Save ucontext_t * across the next call.  */
	mov	r4, r0

	/* __sigprocmask(SIG_BLOCK, NULL, &(ucontext->uc_sigmask)) */
	mov     r0, #SIG_BLOCK
	mov     r1, #0
	add     r2, r4, #UCONTEXT_SIGMASK
	bl      PLTJMP(__sigprocmask)

	/* Store FP regs.  Much of the FP code is copied from arm/setjmp.S.  */

#ifdef PIC
	ldr     r2, 1f
	ldr     r1, .Lrtld_global_ro
0:      add     r2, pc, r2
	ldr     r2, [r2, r1]
	ldr     r2, [r2, #RTLD_GLOBAL_RO_DL_HWCAP_OFFSET]
#else
	ldr     r2, .Lhwcap
	ldr     r2, [r2, #0]
#endif

	add	r0, r4, #UCONTEXT_REGSPACE

#ifdef __SOFTFP__
	tst     r2, #HWCAP_ARM_VFP
	beq     .Lno_vfp
#endif

	/* Store the VFP registers.
	   Don't use VFP instructions directly because this code
	   is used in non-VFP multilibs.  */
	/* Following instruction is vstmia r0!, {d8-d15}.  */
	stc     p11, cr8, [r0], #64
	/* Store the floating-point status register.  */
	/* Following instruction is vmrs r1, fpscr.  */
	mrc     p10, 7, r1, cr1, cr0, 0
	str     r1, [r0], #4
.Lno_vfp:

	tst     r2, #HWCAP_ARM_IWMMXT
	beq     .Lno_iwmmxt

	/* Save the call-preserved iWMMXt registers.  */
	/* Following instructions are wstrd wr10, [r0], #8 (etc.)  */
	stcl    p1, cr10, [r0], #8
	stcl    p1, cr11, [r0], #8
	stcl    p1, cr12, [r0], #8
	stcl    p1, cr13, [r0], #8
	stcl    p1, cr14, [r0], #8
	stcl    p1, cr15, [r0], #8
.Lno_iwmmxt:

	/* Restore the clobbered R4 and LR.  */
	ldr	r14, [r4, #MCONTEXT_ARM_LR]
	ldr	r4, [r4, #MCONTEXT_ARM_R4]

	mov	r0, #0

	DO_RET(r14)

END(__getcontext)

#ifdef PIC
1:      .long   _GLOBAL_OFFSET_TABLE_ - 0b - PC_OFS
.Lrtld_global_ro:
	.long   C_SYMBOL_NAME(_rtld_global_ro)(GOT)
#else
.Lhwcap:
	.long   C_SYMBOL_NAME(_dl_hwcap)
#endif


weak_alias(__getcontext, getcontext)
