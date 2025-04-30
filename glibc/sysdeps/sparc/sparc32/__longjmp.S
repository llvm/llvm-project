/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

ENTRY(__longjmp)
	/* Store our arguments in global registers so we can still
	   use them while unwinding frames and their register windows.  */

	ld ENV(o0,JB_FP), %g3	/* Cache target FP in register %g3.  */
#ifdef PTR_DEMANGLE
	PTR_DEMANGLE (%g3, %g3, %g4)
#endif
	mov %o0, %g1		/* ENV in %g1 */
	orcc %o1, %g0, %g2	/* VAL in %g2 */
	be,a 0f			/* Branch if zero; else skip delay slot.  */
	 mov 1, %g2		/* Delay slot only hit if zero: VAL = 1.  */
0:
	xor %fp, %g3, %o0
	add %fp, 512, %o1
	andncc %o0, 4095, %o0
	bne LOC(thread)
	 cmp %o1, %g3
	bl LOC(thread)

	/* Now we will loop, unwinding the register windows up the stack
	   until the restored %fp value matches the target value in %g3.  */

LOC(loop):
	cmp %fp, %g3		/* Have we reached the target frame? */
	bl,a LOC(loop)		/* Loop while current fp is below target.  */
	 restore		/* Unwind register window in delay slot.  */
	be,a LOC(found)		/* Better have hit it exactly.  */
	 ld ENV(g1,JB_SP), %o0	/* Delay slot: extract target SP.  */

LOC(thread):
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
	ld	ENV(g1,JB_PC), %g5 /* Set return PC. */
	ld	ENV(g1,JB_SP), %g1 /* Set saved SP on restore below. */
	PTR_DEMANGLE2 (%i7, %g5, %g4)
	PTR_DEMANGLE2 (%fp, %g1, %g4)
#else
	ld	ENV(g1,JB_PC), %i7 /* Set return PC. */
	ld	ENV(g1,JB_SP), %fp /* Set saved SP on restore below. */
#endif
	jmp	%i7 + 8
	 restore %g2, 0, %o0	/* Restore values from above register frame. */

LOC(found):
	/* We have unwound register windows so %fp matches the target.  */
#ifdef PTR_DEMANGLE
	PTR_DEMANGLE2 (%sp, %o0, %g4)
#else
	mov %o0, %sp		/* OK, install new SP.  */
#endif

LOC(sp_ok):
	ld ENV(g1,JB_PC), %o0	/* Extract target return PC.  */
#ifdef PTR_DEMANGLE
	PTR_DEMANGLE2 (%o0, %o0, %g4)
#endif
	jmp %o0 + 8		/* Return there.  */
	 mov %g2, %o0		/* Delay slot: set return value.  */

END(__longjmp)
