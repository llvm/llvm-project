/* Copyright (C) 2009-2021 Free Software Foundation, Inc.
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

/* longjmp is implemented in terms of the setcontext trap on Linux/Sparc64.  */

#include <sysdep.h>

/* Offsets into the jmp_buf structure.  */

#define O_mask_was_saved	512
#define O_gregs			32
#define O_g1			(O_gregs + 4*8)
#define O_sp			(O_gregs + 17*8)

.section .rodata.str1.1,"aMS",@progbits,1
	.type	longjmp_msg,@object
longjmp_msg:
	.string "longjmp causes uninitialized stack frame"
	.size	longjmp_msg, .-longjmp_msg

	.text
ENTRY (____longjmp_chk)
	ldx	[%o0 + O_sp], %o2
	cmp	%sp, %o2
	bleu,pt	%xcc, .Lok
	 nop

	save	%sp, -208, %sp
	cfi_remember_state
	cfi_def_cfa_register(%fp)
	cfi_window_save
	cfi_register(%o7, %i7)
	add	%fp, 2023, %o1
	clr	%o0
	LOADSYSCALL(sigaltstack)
	ta	0x6d
	bcs,pn	%xcc, .Lok2
	 lduw	[%fp + 2031], %l2
	andcc	%l2, 0x1, %g0
	be,pn	%xcc, .Lfail
	 ldx	[%fp + 2023], %l0
	ldx	[%fp + 2039], %l1
	sub	%l0, STACK_BIAS, %l0
	add	%l0, %l1, %l0
	sub	%l0, %i2, %l0
	cmp	%l0, %l1
	bgeu,pt	%xcc, .Lok2
	 nop

.Lfail:
#ifndef PIC
	sethi	%hi(longjmp_msg), %o0
	or	%o0, %lo(longjmp_msg), %o0
#else
	SETUP_PIC_REG(l7)
	sethi	%gdop_hix22(longjmp_msg), %o0
	xor	%o0, %gdop_lox10(longjmp_msg), %o0
	ldx	[%l7 + %o0], %o0, %gdop(longjmp_msg)
#endif
	call	HIDDEN_JUMPTARGET(__fortify_fail)
	 nop

.Lok2:	restore
	cfi_restore_state

.Lok:
	/* Modify the context with the value we want to return.  */
	movre	%o1, 1, %o1
	stx	%o1, [%o0 + O_g1]

	/* Let setcontext know if we want to modify the current sigmask. */
	ld	[%o0 + O_mask_was_saved], %o1

	/* And bamf back to where we belong!  */
	ta	0x6f
END(____longjmp_chk)
