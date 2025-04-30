/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* Common bits for implementing software divide.  */

#include <sysdep.h>
#ifdef __linux__
# include <asm/gentrap.h>
# include <asm/pal.h>
#else
# include <machine/pal.h>
#endif

/* These are not normal C functions.  Argument registers are t10 and t11;
   the result goes in t12; the return address is in t9.  Only t12 and AT
   may be clobbered.  */
#define X	t10
#define Y	t11
#define RV	t12
#define RA	t9

/* The secureplt format does not allow the division routines to be called
   via plt; there aren't enough registers free to be clobbered.  Avoid
   setting the symbol type to STT_FUNC, so that the linker won't be tempted
   to create a plt entry.  */
#define funcnoplt notype

/* None of these functions should use implicit anything.  */
	.set	nomacro
	.set	noat

/* Code fragment to invoke _mcount for profiling.  This should be invoked
   directly after allocation of the stack frame.  */
.macro CALL_MCOUNT
#ifdef PROF
	stq	ra, 0(sp)
	stq	pv, 8(sp)
	stq	gp, 16(sp)
	cfi_rel_offset (ra, 0)
	cfi_rel_offset (pv, 8)
	cfi_rel_offset (gp, 16)
	br	AT, 1f
	.set	macro
1:	ldgp	gp, 0(AT)
	mov	RA, ra
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.set	nomacro
	ldq	ra, 0(sp)
	ldq	pv, 8(sp)
	ldq	gp, 16(sp)
	cfi_restore (ra)
	cfi_restore (pv)
	cfi_restore (gp)
	/* Realign subsequent code with what we'd have without this
	   macro at all.  This means aligned with one arithmetic insn
	   used within the bundle.  */
	.align	4
	nop
#endif
.endm

/* In order to make the below work, all top-level divide routines must
   use the same frame size.  */
#define FRAME	64

/* Code fragment to generate an integer divide-by-zero fault.  When
   building libc.so, we arrange for there to be one copy of this code
   placed late in the dso, such that all branches are forward.  When
   building libc.a, we use multiple copies to avoid having an out of
   range branch.  Users should jump to DIVBYZERO.  */

.macro DO_DIVBYZERO
#ifdef PIC
#define DIVBYZERO	__divbyzero
	.section .gnu.linkonce.t.divbyzero, "ax", @progbits
	.globl	__divbyzero
	.type	__divbyzero, @function
	.usepv	__divbyzero, no
	.hidden	__divbyzero
#else
#define DIVBYZERO	$divbyzero
#endif

	.align	4
DIVBYZERO:
	cfi_startproc
	cfi_return_column (RA)
	cfi_def_cfa_offset (FRAME)

	mov	a0, RV
	unop
	lda	a0, GEN_INTDIV
	call_pal PAL_gentrap

	mov	RV, a0
	clr	RV
	lda	sp, FRAME(sp)
	cfi_def_cfa_offset (0)
	ret	$31, (RA), 1

	cfi_endproc
	.size	DIVBYZERO, .-DIVBYZERO
.endm

/* Like the ev6 instructions, but fall back to stack use on prior machines.  */

	.arch	ev6

.macro _ITOFS  gr, fr, slot
#ifdef __alpha_fix__
	itofs	\gr, \fr
#else
	stl	\gr, \slot(sp)
	lds	\fr, \slot(sp)
#endif
.endm

.macro _ITOFT  gr, fr, slot
#ifdef __alpha_fix__
	itoft	\gr, \fr
#else
	stq	\gr, \slot(sp)
	ldt	\fr, \slot(sp)
#endif
.endm

.macro _FTOIT  fr, gr, slot
#ifdef __alpha_fix__
	ftoit	\fr, \gr
#else
	stt	\fr, \slot(sp)
	ldq	\gr, \slot(sp)
#endif
.endm

/* Similarly, but move two registers.  Schedules better for pre-ev6.  */

.macro _ITOFT2 gr1, fr1, slot1, gr2, fr2, slot2
#ifdef __alpha_fix__
	itoft	\gr1, \fr1
	itoft	\gr2, \fr2
#else
	stq	\gr1, \slot1(sp)
	stq	\gr2, \slot2(sp)
	ldt	\fr1, \slot1(sp)
	ldt	\fr2, \slot2(sp)
#endif
.endm
