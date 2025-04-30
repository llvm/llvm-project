/* Install given context.
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
#include <libc-symbols.h>

#include "ucontext_i.h"


ENTRY(__setcontext)
	/* Prologue */
	stw	%r2, -20(%sp)
	.cfi_offset 2, -20
	stwm	%r3, 64(%sp)
	.cfi_def_cfa_offset -64
	.cfi_offset 3, 0
#ifdef PIC
	stw	%r19, -32(%sp)
	.cfi_offset 19, 32
#endif

	/* Save ucp.  */
	copy	%r26, %r3

.Lagain:
	/* Set the current signal mask.  */
	/* sigprocmask(SIG_BLOCK, &ucp->uc_sigmask, NULL);  */
	copy	%r0, %r24
	ldo	oSIGMASK(%r3), %r25
	bl	__sigprocmask, %r2
	ldi	SIG_SETMASK, %r26

	comib,<>,n 0,%ret0,.Lerror

	/* Save %sp, %dp.  */
	copy	%sp, %r4
	copy	%dp, %r5
	copy	%r19, %r6

	/* Get the registers.  */
	ldw	oR1(%r3), %r1
	ldw	oR2(%r3), %r2
	/* ldw	oR3(%r3), %r3 - used for ucp pointer.	*/
	/* ldw	oR4(%r3), %r4 - used for original %sp.	*/
	/* ldw	oR5(%r3), %r5 - used for %dp / %r27.	*/
	/* ldw	oR6(%r3), %r6 - used for %r19.		*/
	ldw	oR7(%r3), %r7
	ldw	oR8(%r3), %r8
	ldw	oR9(%r3), %r9
	ldw	oR10(%r3), %r10
	ldw	oR11(%r3), %r11
	ldw	oR12(%r3), %r12
	ldw	oR13(%r3), %r13
	ldw	oR14(%r3), %r14
	ldw	oR15(%r3), %r15
	ldw	oR16(%r3), %r16
	ldw	oR17(%r3), %r17
	ldw	oR18(%r3), %r18
	ldw	oR19(%r3), %r19
	ldw	oR20(%r3), %r20
	ldw	oR21(%r3), %r21
	/* ldw	oR22(%r3), %r22 - dyncall arg.  */
	ldw	oR23(%r3), %r23
	ldw	oR24(%r3), %r24
	ldw	oR25(%r3), %r25
	ldw	oR26(%r3), %r26
	ldw	oR27(%r3), %r27
	ldw	oR28(%r3), %r28
	ldw	oR29(%r3), %r29
	ldw	oR30(%r3), %sp
	/* ldw	oR31(%r3), %r31 - dyncall scratch register */

	/* Restore floating-point registers.  */
	ldo	 oFPREGS31(%r3), %r22
	fldds	  0(%r22), %fr31
	fldds,mb -8(%r22), %fr30
	fldds,mb -8(%r22), %fr29
	fldds,mb -8(%r22), %fr28
	fldds,mb -8(%r22), %fr27
	fldds,mb -8(%r22), %fr26
	fldds,mb -8(%r22), %fr25
	fldds,mb -8(%r22), %fr24
	fldds,mb -8(%r22), %fr23
	fldds,mb -8(%r22), %fr22
	fldds,mb -8(%r22), %fr21
	fldds,mb -8(%r22), %fr20
	fldds,mb -8(%r22), %fr19
	fldds,mb -8(%r22), %fr18
	fldds,mb -8(%r22), %fr17
	fldds,mb -8(%r22), %fr16
	fldds,mb -8(%r22), %fr15
	fldds,mb -8(%r22), %fr14
	fldds,mb -8(%r22), %fr13
	fldds,mb -8(%r22), %fr12
	fldds,mb -8(%r22), %fr11
	fldds,mb -8(%r22), %fr10
	fldds,mb -8(%r22), %fr9
	fldds,mb -8(%r22), %fr8
	fldds,mb -8(%r22), %fr7
	fldds,mb -8(%r22), %fr6
	fldds,mb -8(%r22), %fr5
	fldds,mb -8(%r22), %fr4
	fldds,mb -8(%r22), %fr3
	fldds,mb -8(%r22), %fr2
	fldds,mb -8(%r22), %fr1
	fldds,mb -8(%r22), %fr0

	/* Do not load oSS_SP into %sp. The value of oSS_SP indicates
	   the start of the user allocated stack, but not the sp that
	   should be used by the new context. In fact makecontext
	   will create a frame, and adjust sp as required. We do not
	   support calling getcontext and modifying ss_sp without
	   a call to makecontext to synchronize ss_sp into the machine
	   context.  */

	/* Call external function.  */
	copy	%r2, %r22
	bl	$$dyncall, %r31
	copy	%r31, %r2

	/* We return here. Get new ucp in %r3, reload %sp.  */
	ldw	oUC_LINK(%r3), %r3
	copy	%r4, %sp
	copy	%r5, %dp
	copy	%r6, %r19

	/* Continue until ucp == NULL.  */
	comib,<> 0,%r3,.Lagain
	nop

	/* No further context available. Exit now.  */
	bl	HIDDEN_JUMPTARGET(exit), %r2
	ldi	0, %r26


.Lerror:
	/* Epilogue */
	ldw	-84(%r30), %r2
#ifdef PIC
	ldw	-32(%r30), %r19
#endif
	bv	%r0(%r2)
	ldwm	-64(%r30), %r3
L(pseudo_end):
PSEUDO_END(__setcontext)

weak_alias(__setcontext, setcontext)
