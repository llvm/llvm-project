/* Set current context.
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

/* int setcontext (const ucontext_t *ucp) */

	.text
	.set	nomips16
LOCALSZ = 0
ARGSZ = 0
MASK = 0x00000000
#ifdef __PIC__
LOCALSZ = 1						/* save gp */
#endif
#if _MIPS_SIM != _ABIO32
ARGSZ = 1						/* save a0 */
# ifdef __PIC__
MASK = 0x10000000
# endif
#endif
FRAMESZ = (((ARGSZ + LOCALSZ) * SZREG) + ALSZ) & ALMASK
GPOFF = FRAMESZ - ((ARGSZ + 1) * SZREG)
#if _MIPS_SIM != _ABIO32
A0OFF = FRAMESZ - (1 * SZREG)				/* callee-allocated */
#else
A0OFF = FRAMESZ + (0 * SZREG)				/* caller-allocated */
#endif
MCONTEXT_GREGSZ = 8
#if _MIPS_SIM == _ABIO32 && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
MCONTEXT_GREGOFF = 4
#else
MCONTEXT_GREGOFF = 0
#endif

NESTED (__setcontext, FRAMESZ, ra)
	.mask	MASK, -(ARGSZ * SZREG)
	.fmask	0x00000000, 0

#ifdef __PIC__
	SETUP_GP
#endif

	PTR_ADDIU sp, -FRAMESZ
	cfi_adjust_cfa_offset (FRAMESZ)

#ifdef __PIC__
	SETUP_GP64_STACK (GPOFF, __setcontext)
	SAVE_GP (GPOFF)
#endif

#ifdef PROF
	.set	noat
	move	AT, ra
	jal	_mcount
	.set	at
#endif

	REG_S	a0, A0OFF(sp)

/* rt_sigprocmask (SIG_SETMASK, &ucp->uc_sigmask, NULL, _NSIG8) */
	li	a3, _NSIG8
	move	a2, zero
	PTR_ADDU a1, a0, UCONTEXT_SIGMASK
	li	a0, SIG_SETMASK

	li	v0, SYS_ify (rt_sigprocmask)
	syscall
	bnez	a3, 99f

	REG_L	v0, A0OFF(sp)

#ifdef __mips_hard_float
# if _MIPS_SIM == _ABI64
	l.d	fs0, (24 * SZREG + MCONTEXT_FPREGS)(v0)
	l.d	fs1, (25 * SZREG + MCONTEXT_FPREGS)(v0)
	l.d	fs2, (26 * SZREG + MCONTEXT_FPREGS)(v0)
	l.d	fs3, (27 * SZREG + MCONTEXT_FPREGS)(v0)
	l.d	fs4, (28 * SZREG + MCONTEXT_FPREGS)(v0)
	l.d	fs5, (29 * SZREG + MCONTEXT_FPREGS)(v0)
	l.d	fs6, (30 * SZREG + MCONTEXT_FPREGS)(v0)
	l.d	fs7, (31 * SZREG + MCONTEXT_FPREGS)(v0)

# else  /* _MIPS_SIM != _ABI64 */
	l.d	fs0, (20 * SZREG + MCONTEXT_FPREGS)(v0)
	l.d	fs1, (22 * SZREG + MCONTEXT_FPREGS)(v0)
	l.d	fs2, (24 * SZREG + MCONTEXT_FPREGS)(v0)
	l.d	fs3, (26 * SZREG + MCONTEXT_FPREGS)(v0)
	l.d	fs4, (28 * SZREG + MCONTEXT_FPREGS)(v0)
	l.d	fs5, (30 * SZREG + MCONTEXT_FPREGS)(v0)

# endif /* _MIPS_SIM != _ABI64 */

	lw	v1, MCONTEXT_FPC_CSR(v0)
	ctc1	v1, fcr31
#endif /* __mips_hard_float */

	/* Note the contents of argument registers will be random
	   unless makecontext() has been called.  */
	REG_L	a0, (MCONTEXT_GREGOFF + 4 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
	REG_L	a1, (MCONTEXT_GREGOFF + 5 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
	REG_L	a2, (MCONTEXT_GREGOFF + 6 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
	REG_L	a3, (MCONTEXT_GREGOFF + 7 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
#if _MIPS_SIM != _ABIO32
	REG_L	a4, (MCONTEXT_GREGOFF + 8 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
	REG_L	a5, (MCONTEXT_GREGOFF + 9 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
	REG_L	a6, (MCONTEXT_GREGOFF + 10 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
	REG_L	a7, (MCONTEXT_GREGOFF + 11 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
#endif

	REG_L	s0, (MCONTEXT_GREGOFF + 16 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
	REG_L	s1, (MCONTEXT_GREGOFF + 17 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
	REG_L	s2, (MCONTEXT_GREGOFF + 18 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
	REG_L	s3, (MCONTEXT_GREGOFF + 19 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
	REG_L	s4, (MCONTEXT_GREGOFF + 20 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
	REG_L	s5, (MCONTEXT_GREGOFF + 21 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
	REG_L	s6, (MCONTEXT_GREGOFF + 22 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
	REG_L	s7, (MCONTEXT_GREGOFF + 23 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
#if ! defined (__PIC__) || _MIPS_SIM != _ABIO32
	REG_L	gp, (MCONTEXT_GREGOFF + 28 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
#endif
	REG_L	sp, (MCONTEXT_GREGOFF + 29 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
	REG_L	fp, (MCONTEXT_GREGOFF + 30 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
	REG_L	ra, (MCONTEXT_GREGOFF + 31 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(v0)
	REG_L	t9, (MCONTEXT_GREGOFF + MCONTEXT_PC)(v0)

	move	v0, zero
	jr	t9

99:
#ifdef __PIC__
	PTR_LA	t9, JUMPTARGET (__syscall_error)
	RESTORE_GP64_STACK
	PTR_ADDIU sp, FRAMESZ
	cfi_adjust_cfa_offset (-FRAMESZ)
	jr	t9

#else  /* ! __PIC__ */

	j	JUMPTARGET (__syscall_error)
#endif /* ! __PIC__ */
PSEUDO_END (__setcontext)

weak_alias (__setcontext, setcontext)
