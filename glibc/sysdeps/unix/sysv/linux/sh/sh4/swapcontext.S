/* Save current context and install the given one.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

#include "ucontext_i.h"

/*  int __swapcontext (ucontext_t *ouc, const ucontext_t *uc);  */

	.text
	.align  5
ENTRY(__swapcontext)

	/* Return value of getcontext.  R0 is the only register whose
	   value is not preserved.  */
	mov	#0, r0
	mov.l	r0, @(oR0,r4)
	mov.l	r1, @(oR1,r4)
	mov.l	r2, @(oR2,r4)
	mov.l	r3, @(oR3,r4)
	mov.l	r4, @(oR4,r4)
	mov.l	r5, @(oR5,r4)
	mov.l	r6, @(oR6,r4)
	mov.l	r7, @(oR7,r4)
	mov	r4, r0
	add	#(oMACL+4), r0
	sts.l	macl, @-r0
	sts.l	mach, @-r0
	stc.l	gbr, @-r0

	/* Save T flag to SR.  */
	movt	r1
	mov.l	r1, @-r0
	sts.l	pr, @-r0

	/* The return address of getcontext is the restart pc.  */
	sts.l	pr, @-r0

	mov.l	r15, @-r0
	mov.l	r14, @-r0
	mov.l	r13, @-r0
	mov.l	r12, @-r0
	mov.l	r11, @-r0
	mov.l	r10, @-r0
	mov.l	r9, @-r0
	mov.l	r8, @-r0

#ifdef __SH_FPU_ANY__
	mov	r4, r0
	/* We need 2 add instruction because oFPUL+4 >= 127.  */
	add	#124,r0
	add	#(oFPUL+4-124),r0
	sts.l	fpul, @-r0
	sts.l	fpscr, @-r0
	frchg
	fmov.s	fr15, @-r0
	fmov.s	fr14, @-r0
	fmov.s	fr13, @-r0
	fmov.s	fr12, @-r0
	fmov.s	fr11, @-r0
	fmov.s	fr10, @-r0
	fmov.s	fr9, @-r0
	fmov.s	fr8, @-r0
	fmov.s	fr7, @-r0
	fmov.s	fr6, @-r0
	fmov.s	fr5, @-r0
	fmov.s	fr4, @-r0
	fmov.s	fr3, @-r0
	fmov.s	fr2, @-r0
	fmov.s	fr1, @-r0
	fmov.s	fr0, @-r0
	frchg
	fmov.s	fr15, @-r0
	fmov.s	fr14, @-r0
	fmov.s	fr13, @-r0
	fmov.s	fr12, @-r0
	fmov.s	fr11, @-r0
	fmov.s	fr10, @-r0
	fmov.s	fr9, @-r0
	fmov.s	fr8, @-r0
	fmov.s	fr7, @-r0
	fmov.s	fr6, @-r0
	fmov.s	fr5, @-r0
	fmov.s	fr4, @-r0
	fmov.s	fr3, @-r0
	fmov.s	fr2, @-r0
	fmov.s	fr1, @-r0
	fmov.s	fr0, @-r0
#endif /* __SH_FPU_ANY__ */

	mov	r5, r8

	/* sigprocmask (SIG_SETMASK, &uc->uc_sigmask, &ouc->uc_sigmask).  */
	mov	#oSIGMASK, r1
	extu.b	r1, r1
	add	r1, r5
	mov	r4, r6
	add	r1, r6
	mov	#SIG_SETMASK, r4
	mov	#+SYS_ify(sigprocmask), r3
	trapa	#0x13
	mov     r0, r1
	mov	#-12, r2
	shad	r2, r1
	not	r1, r1			// r1=0 means r0 = -1 to -4095
	tst	r1, r1			// i.e. error in linux
	bf	.Lswapcontext_restore
.Lsyscall_error:
	SYSCALL_ERROR_HANDLER
.Lpseudo_end:
	rts
	 nop
.Lswapcontext_restore:
#ifdef __SH_FPU_ANY__
	mov	r8, r0
	add	#(oFR0),r0
	fmov.s	@r0+, fr0
	fmov.s	@r0+, fr1
	fmov.s	@r0+, fr2
	fmov.s	@r0+, fr3
	fmov.s	@r0+, fr4
	fmov.s	@r0+, fr5
	fmov.s	@r0+, fr6
	fmov.s	@r0+, fr7
	fmov.s	@r0+, fr8
	fmov.s	@r0+, fr9
	fmov.s	@r0+, fr10
	fmov.s	@r0+, fr11
	fmov.s	@r0+, fr12
	fmov.s	@r0+, fr13
	fmov.s	@r0+, fr14
	fmov.s	@r0+, fr15
	frchg
	fmov.s	@r0+, fr0
	fmov.s	@r0+, fr1
	fmov.s	@r0+, fr2
	fmov.s	@r0+, fr3
	fmov.s	@r0+, fr4
	fmov.s	@r0+, fr5
	fmov.s	@r0+, fr6
	fmov.s	@r0+, fr7
	fmov.s	@r0+, fr8
	fmov.s	@r0+, fr9
	fmov.s	@r0+, fr10
	fmov.s	@r0+, fr11
	fmov.s	@r0+, fr12
	fmov.s	@r0+, fr13
	fmov.s	@r0+, fr14
	fmov.s	@r0+, fr15
	frchg
	lds.l	@r0+, fpscr
	lds.l	@r0+, fpul
#endif /* __SH_FPU_ANY__ */

	mov	r8, r0
	add	#(oPC), r0
	mov.l	@r0+, r2
	lds.l	@r0+, pr

	/* Restore T frag.  */
	mov.l	@r0+, r1
	shlr	r1
	/* Skip GBR which is used for thread pointer.  */
	add	#4, r0

	lds.l	@r0+, mach
	lds.l	@r0+, macl

	mov	r8, r0
	add	#(oR9), r0
	mov.l	@r0+, r9
	mov.l	@r0+, r10
	mov.l	@r0+, r11
	mov.l	@r0+, r12
	mov.l	@r0+, r13
	mov.l	@r0+, r14
	mov.l	@r0+, r15

	mov	r8, r0
	mov.l	@(oR0,r0), r1
	mov.l	r1, @-r15
	cfi_adjust_cfa_offset(4)
	cfi_rel_offset (r1, 0)
	mov.l	r2, @-r15
	cfi_adjust_cfa_offset(4)
	cfi_rel_offset (r2, 0)

	mov.l	@(oR1,r0), r1
	mov.l	@(oR2,r0), r2
	mov.l	@(oR3,r0), r3
	mov.l	@(oR4,r0), r4
	mov.l	@(oR5,r0), r5
	mov.l	@(oR6,r0), r6
	mov.l	@(oR7,r0), r7
	mov.l	@(oR8,r0), r8
	mov.l	@r15+, r0
	cfi_adjust_cfa_offset(-4)
	jmp	@r0
	 mov.l	@r15+, r0

PSEUDO_END(__swapcontext)

weak_alias (__swapcontext, swapcontext)
