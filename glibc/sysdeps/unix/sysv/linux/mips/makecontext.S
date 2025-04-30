/* Modify saved context.
   Copyright (C) 2009-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Maciej W. Rozycki <macro@codesourcery.com>.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>
#include <sys/asm.h>
#include <sys/fpregdef.h>
#include <sys/regdef.h>

#include "ucontext_i.h"

/* int makecontext (ucontext_t *ucp, (void *func) (), int argc, ...) */

	.text
	.set	nomips16
LOCALSZ = 0
ARGSZ = 0
MASK = 0x00000000
#ifdef __PIC__
LOCALSZ = 1						/* save gp */
#endif
#if _MIPS_SIM != _ABIO32
ARGSZ = 5						/* save a3-a7 */
# ifdef __PIC__
MASK = 0x10000000
# endif
#endif
FRAMESZ = (((ARGSZ + LOCALSZ) * SZREG) + ALSZ) & ALMASK
GPOFF = FRAMESZ - ((ARGSZ + 1) * SZREG)
#if _MIPS_SIM != _ABIO32
A3OFF = FRAMESZ - (5 * SZREG)				/* callee-allocated */
A4OFF = FRAMESZ - (4 * SZREG)
A5OFF = FRAMESZ - (3 * SZREG)
A6OFF = FRAMESZ - (2 * SZREG)
A7OFF = FRAMESZ - (1 * SZREG)
NARGREGS = 8
#else
A3OFF = FRAMESZ + (3 * SZREG)				/* caller-allocated */
NARGREGS = 4
#endif
MCONTEXT_GREGSZ = 8
#if _MIPS_SIM == _ABIO32 && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
MCONTEXT_GREGOFF = 4
#else
MCONTEXT_GREGOFF = 0
#endif

NESTED (__makecontext, FRAMESZ, ra)
	.mask	MASK, -(ARGSZ * SZREG)
	.fmask	0x00000000, 0

98:
#ifdef __PIC__
	SETUP_GP
#endif

	PTR_ADDIU sp, -FRAMESZ
	cfi_adjust_cfa_offset (FRAMESZ)

#ifdef __PIC__
	SETUP_GP64_STACK (GPOFF, __makecontext)
	SAVE_GP (GPOFF)
#endif

#ifdef PROF
	.set	noat
	move	AT, ra
	jal	_mcount
	.set	at
#endif

	/* Store args to be passed.  */
	REG_S	a3, A3OFF(sp)
#if _MIPS_SIM != _ABIO32
	REG_S	a4, A4OFF(sp)
	REG_S	a5, A5OFF(sp)
	REG_S	a6, A6OFF(sp)
	REG_S	a7, A7OFF(sp)
#endif

	/* Set up the stack.  */
	PTR_L	t0, STACK_SP(a0)
	PTR_L	t2, STACK_SIZE(a0)
	PTR_ADDIU t1, sp, A3OFF
	PTR_ADDU t0, t2
	and	t0, ALMASK
	blez	a2, 2f					/* no arguments */

	/* Store register arguments.  */
	PTR_ADDIU t2, a0, MCONTEXT_GREGS + 4 * MCONTEXT_GREGSZ + MCONTEXT_GREGOFF
	move	t3, zero
0:
	addiu	t3, 1
	REG_L	v1, (t1)
	PTR_ADDIU t1, SZREG
	REG_S	v1, (t2)
	PTR_ADDIU t2, MCONTEXT_GREGSZ
	bgeu	t3, a2, 2f				/* all done */
	bltu	t3, NARGREGS, 0b			/* next */

	/* Make room for stack arguments.  */
	PTR_SUBU t2, a2, t3
	PTR_SLL	t2, 3
	PTR_SUBU t0, t2
	and	t0, ALMASK

	/* Store stack arguments.  */
	move	t2, t0
1:
	addiu	t3, 1
	REG_L	v1, (t1)
	PTR_ADDIU t1, SZREG
	REG_S	v1, (t2)
	PTR_ADDIU t2, SZREG
	bltu	t3, a2, 1b				/* next */

2:
#if _MIPS_SIM == _ABIO32
	/* Make room for a0-a3 storage.  */
	PTR_ADDIU t0, -(NARGSAVE * SZREG)
#endif
	PTR_L	v1, UCONTEXT_LINK(a0)
#ifdef __PIC__
	PTR_ADDIU t9, 99f - 98b
#else
	PTR_LA	t9, 99f
#endif
	/* sp */
	REG_S	t0, (MCONTEXT_GREGOFF + 29 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(a0)
	/* s0 */
	REG_S	v1, (MCONTEXT_GREGOFF + 16 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(a0)
#ifdef __PIC__
	/* s1 */
	REG_S	gp, (MCONTEXT_GREGOFF + 17 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(a0)
#endif
	/* ra */
	REG_S	t9, (MCONTEXT_GREGOFF + 31 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(a0)
	REG_S	a1, (MCONTEXT_GREGOFF + MCONTEXT_PC)(a0)

#ifdef __PIC__
	RESTORE_GP64_STACK
	PTR_ADDIU sp, FRAMESZ
	cfi_adjust_cfa_offset (-FRAMESZ)
#endif
	jr	ra

	/* We need to terminate the FDE to stop unwinding if backtrace was
	   called within a context created by makecontext.  */
	cfi_endproc
	nop

99:
#ifdef __PIC__
	move	gp, s1
#endif
	move	a0, zero
	beqz	s0, 0f

	/* setcontext (ucp) */
	move	a0, s0
#ifdef __PIC__
	PTR_LA	t9, JUMPTARGET (__setcontext)
	jalr	t9
# if _MIPS_SIM == _ABIO32
	move	gp, s1
# endif
#else
	jal	JUMPTARGET (__setcontext)
#endif
	move	a0, v0

0:
	/* exit (a0) */
#ifdef __PIC__
	PTR_LA	t9, HIDDEN_JUMPTARGET (exit)
	jalr	t9
#else
	jal	HIDDEN_JUMPTARGET (exit)
#endif

	/* You don't exist, you won't feel anything.  */
1:
	lb	zero, (zero)
	b	1b

	cfi_startproc
PSEUDO_END (__makecontext)

weak_alias (__makecontext, makecontext)
