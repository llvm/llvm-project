/* Save current context.
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

/* int getcontext (ucontext_t *ucp) */

	.text
	.set	nomips16
LOCALSZ = 0
MASK = 0x00000000
#ifdef __PIC__
LOCALSZ = 1						/* save gp */
# if _MIPS_SIM != _ABIO32
MASK = 0x10000000
# endif
#endif
FRAMESZ = ((LOCALSZ * SZREG) + ALSZ) & ALMASK
GPOFF = FRAMESZ - (1 * SZREG)
MCONTEXT_GREGSZ = 8
#if _MIPS_SIM == _ABIO32 && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
MCONTEXT_GREGOFF = 4
#else
MCONTEXT_GREGOFF = 0
#endif

NESTED (__getcontext, FRAMESZ, ra)
	.mask	MASK, 0
	.fmask	0x00000000, 0

#ifdef __PIC__
	SETUP_GP

	move	a2, sp
# define _SP a2

# if _MIPS_SIM != _ABIO32
	move	a3, gp
#  define _GP a3
# endif

	PTR_ADDIU sp, -FRAMESZ
	cfi_adjust_cfa_offset (FRAMESZ)
	SETUP_GP64_STACK (GPOFF, __getcontext)
	SAVE_GP (GPOFF)

#else  /* ! __PIC__ */
# define _SP sp
# define _GP gp

#endif /* ! __PIC__ */

#ifdef PROF
	.set	noat
	move	AT, ra
	jal	_mcount
	.set	at
#endif

	REG_S	s0, (MCONTEXT_GREGOFF + 16 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(a0)
	REG_S	s1, (MCONTEXT_GREGOFF + 17 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(a0)
	REG_S	s2, (MCONTEXT_GREGOFF + 18 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(a0)
	REG_S	s3, (MCONTEXT_GREGOFF + 19 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(a0)
	REG_S	s4, (MCONTEXT_GREGOFF + 20 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(a0)
	REG_S	s5, (MCONTEXT_GREGOFF + 21 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(a0)
	REG_S	s6, (MCONTEXT_GREGOFF + 22 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(a0)
	REG_S	s7, (MCONTEXT_GREGOFF + 23 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(a0)
#if ! defined (__PIC__) || _MIPS_SIM != _ABIO32
	REG_S	_GP, (MCONTEXT_GREGOFF + 28 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(a0)
#endif
	REG_S	_SP, (MCONTEXT_GREGOFF + 29 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(a0)
	REG_S	fp, (MCONTEXT_GREGOFF + 30 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(a0)
	REG_S	ra, (MCONTEXT_GREGOFF + 31 * MCONTEXT_GREGSZ + MCONTEXT_GREGS)(a0)
	REG_S	ra, (MCONTEXT_GREGOFF + MCONTEXT_PC)(a0)

#ifdef __mips_hard_float
# if _MIPS_SIM == _ABI64
	s.d	fs0, (24 * SZREG + MCONTEXT_FPREGS)(a0)
	s.d	fs1, (25 * SZREG + MCONTEXT_FPREGS)(a0)
	s.d	fs2, (26 * SZREG + MCONTEXT_FPREGS)(a0)
	s.d	fs3, (27 * SZREG + MCONTEXT_FPREGS)(a0)
	s.d	fs4, (28 * SZREG + MCONTEXT_FPREGS)(a0)
	s.d	fs5, (29 * SZREG + MCONTEXT_FPREGS)(a0)
	s.d	fs6, (30 * SZREG + MCONTEXT_FPREGS)(a0)
	s.d	fs7, (31 * SZREG + MCONTEXT_FPREGS)(a0)

# else  /* _MIPS_SIM != _ABI64 */
	s.d	fs0, (20 * SZREG + MCONTEXT_FPREGS)(a0)
	s.d	fs1, (22 * SZREG + MCONTEXT_FPREGS)(a0)
	s.d	fs2, (24 * SZREG + MCONTEXT_FPREGS)(a0)
	s.d	fs3, (26 * SZREG + MCONTEXT_FPREGS)(a0)
	s.d	fs4, (28 * SZREG + MCONTEXT_FPREGS)(a0)
	s.d	fs5, (30 * SZREG + MCONTEXT_FPREGS)(a0)

# endif /* _MIPS_SIM != _ABI64 */

	cfc1	v1, fcr31
	sw	v1, MCONTEXT_FPC_CSR(a0)
#endif /* __mips_hard_float */

/* rt_sigprocmask (SIG_BLOCK, NULL, &ucp->uc_sigmask, _NSIG8) */
	li	a3, _NSIG8
	PTR_ADDU a2, a0, UCONTEXT_SIGMASK
	move	a1, zero
	li	a0, SIG_BLOCK

	li	v0, SYS_ify (rt_sigprocmask)
	syscall
	bnez	a3, 99f
	cfi_remember_state

#ifdef __PIC__
	RESTORE_GP64_STACK
	PTR_ADDIU sp, FRAMESZ
	cfi_adjust_cfa_offset (-FRAMESZ)
#endif
	move	v0, zero
	jr	ra

99:
	cfi_restore_state
#ifdef __PIC__
	PTR_LA	t9, JUMPTARGET (__syscall_error)
	RESTORE_GP64_STACK
	PTR_ADDIU sp, FRAMESZ
	cfi_adjust_cfa_offset (-FRAMESZ)
	jr	t9

#else  /* ! __PIC__ */

	j	JUMPTARGET (__syscall_error)
#endif /* ! __PIC__ */
PSEUDO_END (__getcontext)

weak_alias (__getcontext, getcontext)
