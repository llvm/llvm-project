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

#include <sysdep.h>
#include <jmpbuf-offsets.h>

#define ENV(base,reg) [%base + (reg * 4)]
#define ST_FLUSH_WINDOWS 3

.section .rodata.str1.1,"aMS",@progbits,1
	.type	longjmp_msg,@object
longjmp_msg:
	.string "longjmp causes uninitialized stack frame"
	.size	longjmp_msg, .-longjmp_msg

	.text
ENTRY (____longjmp_chk)
	ld	ENV(o0,JB_SP), %g5
#ifdef PTR_DEMANGLE
	PTR_DEMANGLE (%g5, %g5, %g4)
#endif

	cmp	%sp, %g5
	bleu	.Lok_norestore
	 nop

	save	%sp, -80, %sp
	cfi_remember_state
	cfi_def_cfa_register(%fp)
	cfi_window_save
	cfi_register(%o7, %i7)

	clr	%o0
	add	%sp, 64, %o1
	LOADSYSCALL(sigaltstack)
	ta	0x10
	bcs	.Lok
	 ld	[%sp + 64 + 4], %o2
	andcc	%o2, 0x1, %g0
	be	.Lfail
	 ld	[%sp + 64 + 0], %o0

	ld	[%sp + 64 + 8], %o1
	add	%o0, %o1, %o0
	sub	%o0, %g5, %o0
	cmp	%o0, %o1
	bgeu	.Lok
	 nop

.Lfail:
#ifndef PIC
	sethi	%hi(longjmp_msg), %o0
	or	%o0, %lo(longjmp_msg), %o0
#else
	SETUP_PIC_REG(l7)
	sethi	%gdop_hix22(longjmp_msg), %o0
	xor	%o0, %gdop_lox10(longjmp_msg), %o0
	ld	[%l7 + %o0], %o0, %gdop(longjmp_msg)
#endif
	call	HIDDEN_JUMPTARGET(__fortify_fail)
	 nop

.Lok:
	restore
	cfi_restore_state

.Lok_norestore:
	ld	ENV(o0,JB_FP), %g3	/* Cache target FP in register %g3.  */
#ifdef PTR_DEMANGLE
	PTR_DEMANGLE2 (%g3, %g3, %g4)
#endif

	mov %o0, %g1		/* ENV in %g1 */
	orcc %o1, %g0, %g2	/* VAL in %g2 */
	be,a 0f			/* Branch if zero; else skip delay slot.  */
	 mov 1, %g2		/* Delay slot only hit if zero: VAL = 1.  */
0:

	save	%sp, -96, %sp
	/*
	 * Do a "flush register windows trap".  The trap handler in the
	 * kernel writes all the register windows to their stack slots, and
	 * marks them all as invalid (needing to be sucked up from the
	 * stack when used).  This ensures that all information needed to
	 * unwind to these callers is in memory, not in the register
	 * windows.
	 */
	ta	ST_FLUSH_WINDOWS
#ifdef PTR_DEMANGLE
	ld	ENV(g1,JB_PC), %g1 /* Set return PC. */
	PTR_DEMANGLE2 (%i7, %g1, %g4)
#else
	ld	ENV(g1,JB_PC), %i7 /* Set return PC. */
#endif
	mov	%g5, %fp
	jmp	%i7 + 8
	 restore %g2, 0, %o0	/* Restore values from above register frame. */

END(____longjmp_chk)
