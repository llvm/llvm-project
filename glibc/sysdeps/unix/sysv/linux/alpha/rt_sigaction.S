/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson <rth@cygnus.com>, 1998

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

/* To enable unwinding through the signal frame without special hackery
   elsewhere, describe the entire struct sigcontext with unwind info.

   In order to minimize the size of the encoding, we set the CFA to the
   end of the sigcontext, which makes all of the registers have small
   negative offsets from that.  */

	.macro SIGCONTEXT_REGS_I base, from=0
	cfi_offset (\from, \base + (4 + \from) * 8)
	.if	30-\from
	SIGCONTEXT_REGS_I \base, "(\from+1)"
	.endif
	.endm

	.macro SIGCONTEXT_REGS_F base, from=32
	cfi_offset (\from, \base + (4 + 1 + \from) * 8)
	.if	62-\from
	SIGCONTEXT_REGS_F \base, "(\from+1)"
	.endif
	.endm

	.macro SIGCONTEXT_REGS base
	SIGCONTEXT_REGS_I \base
	SIGCONTEXT_REGS_F \base
	cfi_offset (63, \base + (4 + 32 + 1 + 32) * 8)
	cfi_offset (64, \base + 2 * 8)
	.endm

	cfi_startproc
	cfi_return_column (64)
	.cfi_signal_frame
	SIGCONTEXT_REGS -648
	cfi_def_cfa_offset (648)

	/* While this frame is marked as a signal frame, that only applies
	   to how this return address is handled for the outer frame.
	   The return address that arrived here, from the inner frame, is
	   not marked as a signal frame and so the unwinder still tries to
	   subtract 1 to examine the presumed call insn.  Thus we must
	   extend the unwind info to a nop before the start.  */
	nop
	.align 4

__syscall_sigreturn:
	mov	sp, a0
	ldi	v0, __NR_sigreturn
	callsys
	.size	__syscall_sigreturn, .-__syscall_sigreturn
	.type	__syscall_sigreturn, @function
	.global __syscall_sigreturn;
	.hidden __syscall_sigreturn;

	/* See above wrt including the nop.  */
	cfi_def_cfa_offset (176 + 648)
	nop
	.align 4

__syscall_rt_sigreturn:
	mov	sp,a0
	ldi	v0,__NR_rt_sigreturn
	callsys
	.size	__syscall_rt_sigreturn, .-__syscall_rt_sigreturn
	.type	__syscall_rt_sigreturn, @function
	.global __syscall_rt_sigreturn;
	.hidden __syscall_rt_sigreturn;

	cfi_endproc
