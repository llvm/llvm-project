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

#include <sysdep.h>
#include "kernel_sysinfo.h"


	.text

ENTRY(__feraiseexcept)
	cfi_startproc
	PSEUDO_PROLOGUE

	lda	sp, -16(sp)
	cfi_adjust_cfa_offset(16)

	ldi	v0, __NR_osf_setsysinfo
	stq	a0, 0(sp)
	mov	sp, a1
	ldi	a0, SSI_IEEE_RAISE_EXCEPTION
	call_pal PAL_callsys

	lda	sp, 16(sp)
	cfi_adjust_cfa_offset(-16)

	/* Here in libm we can't use SYSCALL_ERROR_LABEL.  Nor is it clear
	   that we'd want to set errno anyway.  All we're required to do is
	   return non-zero on error.  Which is exactly A3.  */
	mov	a3, v0
	ret

END(__feraiseexcept)
	cfi_endproc

#if IS_IN (libm)
# include <shlib-compat.h>
# if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_2)
strong_alias (__feraiseexcept, __old_feraiseexcept)
compat_symbol (libm, __old_feraiseexcept, feraiseexcept, GLIBC_2_1);
# endif
libm_hidden_def (__feraiseexcept)
libm_hidden_ver (__feraiseexcept, feraiseexcept)
versioned_symbol (libm, __feraiseexcept, feraiseexcept, GLIBC_2_2);
#endif
