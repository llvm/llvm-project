/* Machine-specific calling sequence for `mcount' profiling function.  x86-64 version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   Contributed by Andreas Jaeger <aj@suse.de>.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* Assembly stub to invoke _mcount().  Compiler generated code calls
   this stub after executing a function's prologue and without saving any
   registers.  It is therefore necessary to preserve %rcx, %rdx, %rsi, %rdi,
   %r8, %r9 as they may contain function arguments.  */

#include <sysdep.h>

ENTRY(_mcount)
	/* Allocate space for 7 registers.  */
	subq	$56,%rsp
	cfi_adjust_cfa_offset (56)
	movq	%rax,(%rsp)
	cfi_rel_offset (rax, 0)
	movq	%rcx,8(%rsp)
	cfi_rel_offset (rcx, 8)
	movq	%rdx,16(%rsp)
	cfi_rel_offset (rdx, 16)
	movq	%rsi,24(%rsp)
	cfi_rel_offset (rsi, 24)
	movq	%rdi,32(%rsp)
	cfi_rel_offset (rdi, 32)
	movq	%r8,40(%rsp)
	cfi_rel_offset (r8, 40)
	movq	%r9,48(%rsp)
	cfi_rel_offset (r9, 48)

	/* Setup parameter for __mcount_internal.  */
	/* selfpc is the return address on the stack.  */
	movq	56(%rsp),%rsi
	/* Get frompc via the frame pointer.  */
	movq	8(%rbp),%rdi
	call C_SYMBOL_NAME(__mcount_internal)
	/* Pop the saved registers.  Please note that `mcount' has no
	   return value.  */
	movq	48(%rsp),%r9
	cfi_restore (r9)
	movq	40(%rsp),%r8
	cfi_restore (r8)
	movq	32(%rsp),%rdi
	cfi_restore (rdi)
	movq	24(%rsp),%rsi
	cfi_restore (rsi)
	movq	16(%rsp),%rdx
	cfi_restore (rdx)
	movq	8(%rsp),%rcx
	cfi_restore (rcx)
	movq	(%rsp),%rax
	cfi_restore (rax)
	addq	$56,%rsp
	cfi_adjust_cfa_offset (-56)
	ret
END(_mcount)

#undef mcount
weak_alias (_mcount, mcount)

/* __fentry__ is different from _mcount in that it is called before
   function prolog.  This means (among other things) that it has non-standard
   stack alignment on entry: (%RSP & 0xF) == 0.  */

ENTRY(__fentry__)
	/* Allocate space for 7 registers
	   (+8 bytes for proper stack alignment).  */
	subq	$64,%rsp
	cfi_adjust_cfa_offset (64)
	movq	%rax,(%rsp)
	cfi_rel_offset (rax, 0)
	movq	%rcx,8(%rsp)
	cfi_rel_offset (rcx, 8)
	movq	%rdx,16(%rsp)
	cfi_rel_offset (rdx, 16)
	movq	%rsi,24(%rsp)
	cfi_rel_offset (rsi, 24)
	movq	%rdi,32(%rsp)
	cfi_rel_offset (rdi, 32)
	movq	%r8,40(%rsp)
	cfi_rel_offset (r8, 40)
	movq	%r9,48(%rsp)
	cfi_rel_offset (r9, 48)

	/* Setup parameter for __mcount_internal.  */
	/* selfpc is the return address on the stack.  */
	movq	64(%rsp),%rsi
	/* caller is the return address above it */
	movq	72(%rsp),%rdi
	call C_SYMBOL_NAME(__mcount_internal)
	/* Pop the saved registers.  Please note that `__fentry__' has no
	   return value.  */
	movq	48(%rsp),%r9
	cfi_restore (r9)
	movq	40(%rsp),%r8
	cfi_restore (r8)
	movq	32(%rsp),%rdi
	cfi_restore (rdi)
	movq	24(%rsp),%rsi
	cfi_restore (rsi)
	movq	16(%rsp),%rdx
	cfi_restore (rdx)
	movq	8(%rsp),%rcx
	cfi_restore (rcx)
	movq	(%rsp),%rax
	cfi_restore (rax)
	addq	$64,%rsp
	cfi_adjust_cfa_offset (-64)
	ret
END(__fentry__)
