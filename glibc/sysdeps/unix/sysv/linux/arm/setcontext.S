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

/* ??? Needs more rearrangement for the LDM to handle thumb mode.  */
#define NO_THUMB
#include <sysdep.h>
#include <rtld-global-offsets.h>

#include "ucontext_i.h"

	.syntax unified
	.text

/* int setcontext (const ucontext_t *ucp) */

ENTRY(__setcontext)
	mov	r4, r0
	add	r0, r0, #UCONTEXT_REGSPACE

	/* Restore the VFP registers.  Copied from arm/__longjmp.S.  */
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

#ifdef __SOFTFP__
	tst     r2, #HWCAP_ARM_VFP
	beq     .Lno_vfp_sc
#endif

	/* Following instruction is vldmia r0!, {d8-d15}.  */
	ldc     p11, cr8, [r0], #64
	/* Restore the floating-point status register.  */
	ldr     r1, [r0], #4
	/* Following instruction is fmxr fpscr, r1.  */
	mcr     p10, 7, r1, cr1, cr0, 0
.Lno_vfp_sc:
	tst     r2, #HWCAP_ARM_IWMMXT
	beq     .Lno_iwmmxt_sc

	/* Restore the call-preserved iWMMXt registers.  */
	/* Following instructions are wldrd wr10, [r0], #8 (etc.)  */
	ldcl    p1, cr10, [r0], #8
	ldcl    p1, cr11, [r0], #8
	ldcl    p1, cr12, [r0], #8
	ldcl    p1, cr13, [r0], #8
	ldcl    p1, cr14, [r0], #8
	ldcl    p1, cr15, [r0], #8
.Lno_iwmmxt_sc:

	/* Now bring back the signal status.  */
	mov	r0, #SIG_SETMASK
	add	r1, r4, #UCONTEXT_SIGMASK
	mov	r2, #0
	bl	PLTJMP(__sigprocmask)

	/* Loading r0-r3 makes makecontext easier.  */
	add     r14, r4, #MCONTEXT_ARM_R0
	ldmia   r14, {r0-r12}
	ldr     r13, [r14, #(MCONTEXT_ARM_SP - MCONTEXT_ARM_R0)]
	add     r14, r14, #(MCONTEXT_ARM_LR - MCONTEXT_ARM_R0)
	ldmia   r14, {r14, pc}

END(setcontext)
weak_alias(__setcontext, setcontext)

	/* Called when a makecontext() context returns.  Start the
	   context in R4 or fall through to exit().  */
	/* Unwind descriptors are looked up based on PC - 2, so we have to
	   make sure to mark the instruction preceding the __startcontext
	   label as .cantunwind.  */
	.fnstart
	.cantunwind
	nop
ENTRY(__startcontext)
	movs    r0, r4
	bne     PLTJMP(__setcontext)

	@ New context was 0 - exit
	b       PLTJMP(HIDDEN_JUMPTARGET(exit))
	.fnend
END(__startcontext)

#ifdef PIC
1:      .long   _GLOBAL_OFFSET_TABLE_ - 0b - PC_OFS
.Lrtld_global_ro:
	.long   C_SYMBOL_NAME(_rtld_global_ro)(GOT)
#else
.Lhwcap:
	.long   C_SYMBOL_NAME(_dl_hwcap)
#endif
