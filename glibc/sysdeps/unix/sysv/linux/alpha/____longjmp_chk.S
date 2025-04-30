/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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
#include <jmpbuf-offsets.h>


	.section .rodata.str1.1,"aMS",@progbits,1
	.type   longjmp_msg,@object
longjmp_msg:
	.string "longjmp causes uninitialized stack frame"
	.size   longjmp_msg, .-longjmp_msg


/* Jump to the position specified by ENV, causing the
   setjmp call there to return VAL, or 1 if VAL is 0.
   void __longjmp (__jmp_buf env, int val).  */
	.text
	.align	4
	.globl	____longjmp_chk
	.type	____longjmp_chk, @function
	.usepv	____longjmp_chk, std

	cfi_startproc
____longjmp_chk:
	ldgp    gp, 0(pv)
#ifdef PROF
	.set noat
	lda     AT, _mcount
	jsr     AT, (AT), _mcount
	.set at
#endif

	ldq     s2, JB_PC*8(a0)
	mov	a0, s0
	ldq     fp, JB_FP*8(a0)
	mov     a1, s1
	ldq     s3, JB_SP*8(a0)
	cmoveq  s1, 1, s1

#ifdef PTR_DEMANGLE
	PTR_DEMANGLE(s2, t1)
	PTR_DEMANGLE2(s3, t1)
	PTR_DEMANGLE2(fp, t1)
#endif
	/* ??? While this is a proper test for detecting a longjmp to an
	   invalid frame within any given stack, the main thread stack is
	   located *below* almost everything in the address space.  Which
	   means that the test at Lfail vs the signal stack will almost
	   certainly never pass.  We ought bounds check top and bottom of
	   the current thread's stack.  */
	cmpule	s3, sp, t1
	bne	t1, $Lfail

	.align	4
$Lok:
	mov	s0, a0
	mov	s1, v0
	mov	s3, t0
	mov	s2, ra
	cfi_remember_state
	cfi_def_cfa(a0, 0)
	cfi_register(sp, t0)
	cfi_offset(s0, JB_S0*8)
	cfi_offset(s1, JB_S1*8)
	cfi_offset(s2, JB_S2*8)
	cfi_offset(s3, JB_S3*8)
	cfi_offset(s4, JB_S4*8)
	cfi_offset(s5, JB_S5*8)
	cfi_offset(s3, JB_S3*8)
	cfi_offset($f2, JB_F2*8)
	cfi_offset($f3, JB_F3*8)
	cfi_offset($f4, JB_F4*8)
	cfi_offset($f5, JB_F5*8)
	cfi_offset($f6, JB_F6*8)
	cfi_offset($f7, JB_F7*8)
	cfi_offset($f8, JB_F8*8)
	cfi_offset($f9, JB_F9*8)
	ldq	s0, JB_S0*8(a0)
	ldq	s1, JB_S1*8(a0)
	ldq	s2, JB_S2*8(a0)
	ldq	s3, JB_S3*8(a0)
	ldq	s4, JB_S4*8(a0)
	ldq	s5, JB_S5*8(a0)
	ldt     $f2, JB_F2*8(a0)
	ldt     $f3, JB_F3*8(a0)
	ldt     $f4, JB_F4*8(a0)
	ldt     $f5, JB_F5*8(a0)
	ldt     $f6, JB_F6*8(a0)
	ldt     $f7, JB_F7*8(a0)
	ldt     $f8, JB_F8*8(a0)
	ldt     $f9, JB_F9*8(a0)
	mov     t0, sp
	ret

	.align	4
$Lfail:
	cfi_restore_state
	lda	v0, __NR_sigaltstack
	lda	a0, 0
	lda	a1, -32(sp)
	lda	sp, -32(sp)
	cfi_adjust_cfa_offset(32)
	callsys
	ldq	t0, 0(sp)	/* ss_sp */
	ldl	t1, 8(sp)	/* ss_flags */
	ldq	t2, 16(sp)	/* ss_size */
	lda	sp, 32(sp)
	cfi_adjust_cfa_offset(-32)

	/* Without working sigaltstack we cannot perform the test.  */
	bne	a3, $Lok

	addq	t0, t2, t0	/* t0 = ss_sp + ss_size */
	subq	t0, s3, t0	/* t0 = (ss_sp + ss_size) - new_sp */
	cmpule	t2, t0, t0	/* t0 = (t0 >= ss_size) */
	and	t0, t1, t0	/* t0 = (t0 >= ss_size) & (ss_flags & SS_ONSTACK) */
	bne	t0, $Lok

	ldah	a0, longjmp_msg(gp)	!gprelhigh
	lda	a0, longjmp_msg(a0)	!gprellow
#ifdef PIC
	jsr	ra, HIDDEN_JUMPTARGET(__fortify_fail)
#else
	bsr	ra, HIDDEN_JUMPTARGET(__fortify_fail)	!samegp
#endif
	bugchk

	cfi_endproc
	.size	____longjmp_chk, .-____longjmp_chk
