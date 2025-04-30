/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

#include <sys/asm.h>

/* Usage:
   long int syscall (syscall_number, arg1, arg2, arg3, arg4, arg5, arg6, arg7)

   We need to do some arg shifting, syscall_number will be in v0.  */


	.text
NESTED (syscall, SZREG, ra)
	.mask 0x00010000, -SZREG
	.fmask 0x00000000, 0
	PTR_ADDIU sp, -SZREG
	cfi_adjust_cfa_offset (SZREG)
	REG_S s0, (sp)
	cfi_rel_offset (s0, 0)

	move s0, a0
	move a0, a1		/* shift arg1 - arg7.  */
	move a1, a2
	move a2, a3
	move a3, a4
	move a4, a5
	move a5, a6
	move a6, a7

	move v0, s0		/* Syscall number -> v0 */
	syscall			/* Do the system call.  */

	REG_L s0, (sp)
	cfi_restore (s0)
	PTR_ADDIU sp, SZREG
	cfi_adjust_cfa_offset (-SZREG)
	bne a3, zero, L(error)

	ret

L(error):
	SETUP_GP64_REG (a0, syscall)
	PTR_LA t9, __syscall_error
	RESTORE_GP64_REG
	jr t9

END (syscall)
