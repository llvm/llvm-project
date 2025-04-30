/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Mosberger <davidm@azstarnet.com>, 1995.

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


ENTRY(__ieee_set_fp_control)
	cfi_startproc
	PSEUDO_PROLOGUE

	lda	sp, -16(sp)
	cfi_adjust_cfa_offset(16)

	ldi	v0, __NR_osf_setsysinfo
	stq	a0, 0(sp)
	mov	sp, a1
	ldi	a0, SSI_IEEE_FP_CONTROL
	call_pal PAL_callsys

	lda	sp, 16(sp)
	cfi_adjust_cfa_offset(-16)

	bne	a3, SYSCALL_ERROR_LABEL
	ret

PSEUDO_END(__ieee_set_fp_control)
	cfi_endproc

libc_hidden_def(__ieee_set_fp_control)
weak_alias (__ieee_set_fp_control, ieee_set_fp_control)
