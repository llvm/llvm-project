/* Copyright (C) 2009-2021 Free Software Foundation, Inc.
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

#include <sigaltstack-offsets.h>

	.section .rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC0:
	.string "longjmp causes uninitialized stack frame"

	.section .sdata,"aws",@progbits
	.align 8
	.type	longjmp_msg,@object
longjmp_msg:
	data8	.LC0
	.size	longjmp_msg, .-longjmp_msg

#define __longjmp ____longjmp_chk

/* We use 32 bytes (rather than sizeof(stack_t)) so that we keep the stack
   properly aligned.  But we still want a sanity check to make sure 32 is
   actually enough.  */
#define STACK_SPACE ((sizeSS + 31) & -32)

/* Check the stack pointer held in the jumpbuf.  Make sure it's in either the
   current stack (r12) or in the signal stack.  */
#define CHECK_RSP						\
	ld8 loc0 = [in0];					\
	;;							\
	/* First see if target stack is within current one.  */	\
	cmp.ltu p0, p8 = loc0, r12;				\
(p8)	br.cond.dptk.many .Lok;					\
								\
	/* Check if it's an alternative signal stack.  */	\
	mov out0 = r0;						\
	add out1 = -STACK_SPACE, r12;				\
	;;							\
	mov r12 = out1;						\
	DO_CALL_VIA_BREAK (SYS_ify (sigaltstack));		\
	;;							\
	/* If the syscall failed, then assume it's OK.  */	\
	cmp.eq p8, p0 = -1, r10;				\
(p8)	br.cond.spnt .Lok;					\
	/* Move stack_t into regs.  */				\
	add r14 = oSS_FLAGS, r12;	/* ss_flags */		\
	add r15 = oSS_SIZE, r12;	/* ss_size */		\
	ld8 r16 = [r12];		/* ss_sp */		\
	;;							\
	ld4 r17 = [r14];		/* ss_flags */		\
	ld8 r18 = [r15];		/* ss_size */		\
	;;							\
	sub r19 = r16, r18;		/* sp - size */		\
	/* See if we're currently on the altstack.  */		\
	tbit.nz p0, p8 = r17, 0;	/* SS_ONSTACK */	\
(p8)	br.cond.spnt .Lfail;					\
	/* Verify target is within alternative stack.  */	\
	cmp.gtu p7, p0 = loc0, r16;				\
(p7)	br.cond.spnt .Lfail;					\
	;;							\
	cmp.ltu p0, p8 = loc0, r19;				\
(p8)	br.cond.sptk.many .Lok;					\
	;;							\
								\
	/* Still here?  Abort!  */				\
.Lfail:								\
	add r12 = STACK_SPACE, r12;				\
	addl loc0 = @ltoffx(longjmp_msg#), r1;;			\
	ld8.mov loc0 = [loc0], longjmp_msg#;;			\
	ld8 out0 = [loc0];					\
	br.call.sptk.many b0 = HIDDEN_JUMPTARGET(__fortify_fail)#;; \
.Lok:								\
	add r12 = STACK_SPACE, r12;

#include "__longjmp.S"
