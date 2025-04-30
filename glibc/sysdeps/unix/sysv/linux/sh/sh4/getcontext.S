/* Save current context.
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

/*  int __getcontext (ucontext_t *uc);  */

	.text
	.align  5
ENTRY(__getcontext)

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
	/* We need 2 add instruction because oFPUL+4 > 127.  */
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

	/* sigprocmask (SIG_BLOCK, NULL, &uc->uc_sigmask).  */
	mov	r4, r6
	/* We need 2 add instruction because oSIGMASK > 127.  */
	add	#(oSIGMASK/2), r6
	add	#(oSIGMASK/2), r6
	mov	#SIG_BLOCK, r4
	mov	#0, r5
	mov	#+SYS_ify(sigprocmask), r3
	trapa	#0x13
	mov     r0, r1
	mov	#-12, r2
	shad	r2, r1
	not	r1, r1			// r1=0 means r0 = -1 to -4095
	tst	r1, r1			// i.e. error in linux
	bf	.Lgetcontext_end
.Lsyscall_error:
	SYSCALL_ERROR_HANDLER
.Lgetcontext_end:
	/* All done, return 0 for success.  */
	mov	#0, r0
.Lpseudo_end:
	rts
	 nop

PSEUDO_END(__getcontext)

weak_alias (__getcontext, getcontext)
