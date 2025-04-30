/* Save current context.
   Copyright (C) 2009-2021 Free Software Foundation, Inc.
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

#include "ucontext-macros.h"

/* int getcontext (ucontext_t *ucp) */

	.text
LEAF (__getcontext)
	SAVE_INT_REG (ra,   0, a0)
	SAVE_INT_REG (ra,   1, a0)
	SAVE_INT_REG (sp,   2, a0)
	SAVE_INT_REG (s0,   8, a0)
	SAVE_INT_REG (s1,   9, a0)
	SAVE_INT_REG (x0,  10, a0)	/* return 0 by overwriting a0.  */
	SAVE_INT_REG (s2,  18, a0)
	SAVE_INT_REG (s3,  19, a0)
	SAVE_INT_REG (s4,  20, a0)
	SAVE_INT_REG (s5,  21, a0)
	SAVE_INT_REG (s6,  22, a0)
	SAVE_INT_REG (s7,  23, a0)
	SAVE_INT_REG (s8,  24, a0)
	SAVE_INT_REG (s9,  25, a0)
	SAVE_INT_REG (s10, 26, a0)
	SAVE_INT_REG (s11, 27, a0)

#ifndef __riscv_float_abi_soft
	frsr	a1

	SAVE_FP_REG (fs0,   8, a0)
	SAVE_FP_REG (fs1,   9, a0)
	SAVE_FP_REG (fs2,  18, a0)
	SAVE_FP_REG (fs3,  19, a0)
	SAVE_FP_REG (fs4,  20, a0)
	SAVE_FP_REG (fs5,  21, a0)
	SAVE_FP_REG (fs6,  22, a0)
	SAVE_FP_REG (fs7,  23, a0)
	SAVE_FP_REG (fs8,  24, a0)
	SAVE_FP_REG (fs9,  25, a0)
	SAVE_FP_REG (fs10, 26, a0)
	SAVE_FP_REG (fs11, 27, a0)

	sw	a1, MCONTEXT_FSR(a0)
#endif /* __riscv_float_abi_soft */

/* rt_sigprocmask (SIG_BLOCK, NULL, &ucp->uc_sigmask, _NSIG8) */
	li	a3, _NSIG8
	add     a2, a0, UCONTEXT_SIGMASK
	mv	a1, zero
	li	a0, SIG_BLOCK

	li	a7, SYS_ify (rt_sigprocmask)
	scall
	bltz	a0, 99f

	ret

99:	j	__syscall_error

PSEUDO_END (__getcontext)

weak_alias (__getcontext, getcontext)
