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


	.text

ENTRY(__ieee_get_fp_control)
	cfi_startproc
	PSEUDO_PROLOGUE

	lda	sp, -16(sp)
	cfi_adjust_cfa_offset(16)

	mov	sp, a1
	ldi	a0, GSI_IEEE_FP_CONTROL
	ldi	v0, __NR_osf_getsysinfo
	call_pal PAL_callsys

	ldq	t0, 0(sp)
	lda	sp, 16(sp)
	cfi_adjust_cfa_offset(-16)

	bne	a3, SYSCALL_ERROR_LABEL

	mov	t0, v0
	ret

PSEUDO_END(__ieee_get_fp_control)
	cfi_endproc

libc_hidden_def(__ieee_get_fp_control)
weak_alias (__ieee_get_fp_control, ieee_get_fp_control)
