/* Get current user context.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Helge Deller <deller@gmx.de>, 2008.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

#include "ucontext_i.h"


	/* Trampoline function. Non-standard calling ABI.  */
	/* Can not use ENTRY(__getcontext_ret) here.  */
	.type	__getcontext_ret, @function
	.hidden	__getcontext_ret
__getcontext_ret:
	.proc
	.callinfo FRAME=0,NO_CALLS
	/* r26-r23 contain original r3-r6, but because setcontext
	   does not reload r3-r6 (it's using them as temporaries)
	   we must save them elsewhere and swap them back in.  */
	copy	%r23, %r3
	copy	%r24, %r4
	copy	%r25, %r5
	copy	%r26, %r6
	/* r20 contains original return pointer.  */
	bv	0(%r20)
	copy	%r0, %ret0
	.procend
	.size	__getcontext_ret, .-__getcontext_ret


ENTRY(__getcontext)
	/* Save the registers.  */
	stw	%r0, oR0(%r26)
	stw	%r1, oR1(%r26)
	/* stw	%r2, oR2(%r26) - used for trampoline.  */
	stw	%r3, oR3(%r26)
	stw	%r4, oR4(%r26)
	stw	%r5, oR5(%r26)
	stw	%r6, oR6(%r26)
	stw	%r7, oR7(%r26)
	stw	%r8, oR8(%r26)
	stw	%r9, oR9(%r26)
	stw	%r10, oR10(%r26)
	stw	%r11, oR11(%r26)
	stw	%r12, oR12(%r26)
	stw	%r13, oR13(%r26)
	stw	%r14, oR14(%r26)
	stw	%r15, oR15(%r26)
	stw	%r16, oR16(%r26)
	stw	%r17, oR17(%r26)
	stw	%r18, oR18(%r26)
	stw	%r19, oR19(%r26)
	/* stw	%r20, oR20(%r26) - used for trampoline.  */
	stw	%r21, oR21(%r26)
	stw	%r22, oR22(%r26)
	/* stw	%r23, oR23(%r26) - used for trampoline.  */
	/* stw	%r24, oR24(%r26) - used for trampoline.  */
	/* stw	%r25, oR25(%r26) - used for trampoline.  */
	/* stw	%r26, oR26(%r26) - used for trampoline.  */
	stw	%r27, oR27(%r26)
	stw	%r28, oR28(%r26)
	stw	%r29, oR29(%r26)
	stw	%sp, oR30(%r26)
	stw	%r31, oR31(%r26)

	stw	%r0, oUC_FLAGS(%r26)
	/* stw	%r0, oUC_LINK(%r26) - Do not overwrite.  */
	stw	%sp, oSS_SP(%r26)
	stw	%r0, oSS_FLAGS(%r26)
	stw	%r0, oSS_SIZE(%r26)

	stw	%r0, oSC_FLAGS(%r26)

	stw	%r0, oIASQ0(%r26)
	stw	%r0, oIASQ1(%r26)
	stw	%r0, oIAOQ0(%r26)
	stw	%r0, oIAOQ1(%r26)
	stw	%r0, oSAR(%r26) /* used as flag in swapcontext().  */


	/* Store floating-point regs.  */
	ldo	oFPREGS0(%r26),%r1
	fstds,ma %fr0, 8(%r1)
	fstds,ma %fr1, 8(%r1)
	fstds,ma %fr2, 8(%r1)
	fstds,ma %fr3, 8(%r1)
	fstds,ma %fr4, 8(%r1)
	fstds,ma %fr5, 8(%r1)
	fstds,ma %fr6, 8(%r1)
	fstds,ma %fr7, 8(%r1)
	fstds,ma %fr8, 8(%r1)
	fstds,ma %fr9, 8(%r1)
	fstds,ma %fr10, 8(%r1)
	fstds,ma %fr11, 8(%r1)
	fstds,ma %fr12, 8(%r1)
	fstds,ma %fr13, 8(%r1)
	fstds,ma %fr14, 8(%r1)
	fstds,ma %fr15, 8(%r1)
	fstds,ma %fr16, 8(%r1)
	fstds,ma %fr17, 8(%r1)
	fstds,ma %fr18, 8(%r1)
	fstds,ma %fr19, 8(%r1)
	fstds,ma %fr20, 8(%r1)
	fstds,ma %fr21, 8(%r1)
	fstds,ma %fr22, 8(%r1)
	fstds,ma %fr23, 8(%r1)
	fstds,ma %fr24, 8(%r1)
	fstds,ma %fr25, 8(%r1)
	fstds,ma %fr26, 8(%r1)
	fstds,ma %fr27, 8(%r1)
	fstds,ma %fr28, 8(%r1)
	fstds,ma %fr29, 8(%r1)
	fstds,ma %fr30, 8(%r1)
	fstds	 %fr31, 0(%r1)

	/* Prologue */
	stw	%r2, -20(%sp)
	.cfi_offset 2, -20
	stwm	%r4, 64(%sp)
	.cfi_def_cfa_offset -64
	.cfi_offset 4, 0
#ifdef PIC
	stw	%r19, -32(%sp)
	.cfi_offset 19, 32
#endif

	/* Set up the trampoline registers.
	   r20, r23, r24, r25, r26 and r2 are clobbered
	   by call to getcontext() anyway. Reuse them.  */
	stw	%r2, oR20(%r26)
	stw	%r3, oR23(%r26)
	stw	%r4, oR24(%r26)
	stw	%r5, oR25(%r26)
	stw	%r6, oR26(%r26)
#ifdef PIC
	addil	LT%__getcontext_ret, %r19
	ldw     RT%__getcontext_ret(%r1), %r1
#else
	ldil	L%__getcontext_ret, %r1
	ldo     R%__getcontext_ret(%r1), %r1
#endif
	stw	%r1, oR2(%r26)

	/* Save the current signal mask.  */
	/* sigprocmask(SIG_BLOCK, NULL, &ucp->uc_sigmask);  */
	ldo	oSIGMASK(%r26), %r24
	copy	%r0, %r25
	bl	__sigprocmask, %r2
	ldi	SIG_BLOCK, %r26

	/* Epilogue */
	ldw	-84(%sp), %r2
#ifdef PIC
	ldw	-32(%sp), %r19
#endif
	bv	%r0(%r2)
	ldwm	-64(%sp), %r4
END(__getcontext)

weak_alias (__getcontext, getcontext)
