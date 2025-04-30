/* Save current context.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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
#include <ucontext-offsets.h>

/* ??? Should be a better place for this that's asm friendly.  */
#define SIG_BLOCK     1


ENTRY (__getcontext)
#ifdef PROF
	ldgp    gp, 0(pv)
	.set noat
	lda     AT, _mcount
	jsr     AT, (AT), _mcount
	.set at
	.prologue 1
#else
	.prologue 0
#endif

	bsr	$0, __getcontext_x
	mov	$31, $0
	ret

END(__getcontext)
weak_alias (__getcontext, getcontext)


/* An internal routine used by getcontext and setcontext.
   The incomming return address register is $0.  */

	.align	4
	.globl	__getcontext_x
	.hidden	__getcontext_x
	.usepv	__getcontext_x, no

	cfi_startproc
	cfi_return_column (64)
__getcontext_x:
	cfi_register (64, 0)

	.set noat

	/* Return value of getcontext.  $0 is the only register
	   whose value is not preserved. */
	stq	$31, UC_SIGCTX+SC_REGS($16)

	/* Store all registers into the context.  */
	stq	$1, UC_SIGCTX+SC_REGS+1*8($16)
	stq	$2, UC_SIGCTX+SC_REGS+2*8($16)
	stq	$3, UC_SIGCTX+SC_REGS+3*8($16)
	stq	$4, UC_SIGCTX+SC_REGS+4*8($16)
	stq	$5, UC_SIGCTX+SC_REGS+5*8($16)
	stq	$6, UC_SIGCTX+SC_REGS+6*8($16)
	stq	$7, UC_SIGCTX+SC_REGS+7*8($16)
	stq	$8, UC_SIGCTX+SC_REGS+8*8($16)
	stq	$9, UC_SIGCTX+SC_REGS+9*8($16)
	stq	$10, UC_SIGCTX+SC_REGS+10*8($16)
	stq	$11, UC_SIGCTX+SC_REGS+11*8($16)
	stq	$12, UC_SIGCTX+SC_REGS+12*8($16)
	stq	$13, UC_SIGCTX+SC_REGS+13*8($16)
	stq	$14, UC_SIGCTX+SC_REGS+14*8($16)
	stq	$15, UC_SIGCTX+SC_REGS+15*8($16)
	stq	$16, UC_SIGCTX+SC_REGS+16*8($16)
	stq	$17, UC_SIGCTX+SC_REGS+17*8($16)
	stq	$18, UC_SIGCTX+SC_REGS+18*8($16)
	stq	$19, UC_SIGCTX+SC_REGS+19*8($16)
	stq	$20, UC_SIGCTX+SC_REGS+20*8($16)
	stq	$21, UC_SIGCTX+SC_REGS+21*8($16)
	stq	$22, UC_SIGCTX+SC_REGS+22*8($16)
	stq	$23, UC_SIGCTX+SC_REGS+23*8($16)
	stq	$24, UC_SIGCTX+SC_REGS+24*8($16)
	stq	$25, UC_SIGCTX+SC_REGS+25*8($16)
	stq	$26, UC_SIGCTX+SC_REGS+26*8($16)
	stq	$27, UC_SIGCTX+SC_REGS+27*8($16)
	stq	$28, UC_SIGCTX+SC_REGS+28*8($16)
	stq	$29, UC_SIGCTX+SC_REGS+29*8($16)
	stq	$30, UC_SIGCTX+SC_REGS+30*8($16)
	stq	$31, UC_SIGCTX+SC_REGS+31*8($16)

	stt	$f0, UC_SIGCTX+SC_FPREGS+0*8($16)
	stt	$f1, UC_SIGCTX+SC_FPREGS+1*8($16)
	stt	$f2, UC_SIGCTX+SC_FPREGS+2*8($16)
	stt	$f3, UC_SIGCTX+SC_FPREGS+3*8($16)
	stt	$f4, UC_SIGCTX+SC_FPREGS+4*8($16)
	stt	$f5, UC_SIGCTX+SC_FPREGS+5*8($16)
	stt	$f6, UC_SIGCTX+SC_FPREGS+6*8($16)
	stt	$f7, UC_SIGCTX+SC_FPREGS+7*8($16)
	stt	$f8, UC_SIGCTX+SC_FPREGS+8*8($16)
	stt	$f9, UC_SIGCTX+SC_FPREGS+9*8($16)
	stt	$f10, UC_SIGCTX+SC_FPREGS+10*8($16)
	stt	$f11, UC_SIGCTX+SC_FPREGS+11*8($16)
	stt	$f12, UC_SIGCTX+SC_FPREGS+12*8($16)
	stt	$f13, UC_SIGCTX+SC_FPREGS+13*8($16)
	stt	$f14, UC_SIGCTX+SC_FPREGS+14*8($16)
	stt	$f15, UC_SIGCTX+SC_FPREGS+15*8($16)
	stt	$f16, UC_SIGCTX+SC_FPREGS+16*8($16)
	stt	$f17, UC_SIGCTX+SC_FPREGS+17*8($16)
	stt	$f18, UC_SIGCTX+SC_FPREGS+18*8($16)
	stt	$f19, UC_SIGCTX+SC_FPREGS+19*8($16)
	stt	$f20, UC_SIGCTX+SC_FPREGS+20*8($16)
	stt	$f21, UC_SIGCTX+SC_FPREGS+21*8($16)
	stt	$f22, UC_SIGCTX+SC_FPREGS+22*8($16)
	stt	$f23, UC_SIGCTX+SC_FPREGS+23*8($16)
	stt	$f24, UC_SIGCTX+SC_FPREGS+24*8($16)
	stt	$f25, UC_SIGCTX+SC_FPREGS+25*8($16)
	stt	$f26, UC_SIGCTX+SC_FPREGS+26*8($16)
	stt	$f27, UC_SIGCTX+SC_FPREGS+27*8($16)
	stt	$f28, UC_SIGCTX+SC_FPREGS+28*8($16)
	stt	$f29, UC_SIGCTX+SC_FPREGS+29*8($16)
	stt	$f30, UC_SIGCTX+SC_FPREGS+30*8($16)
	stt	$f31, UC_SIGCTX+SC_FPREGS+31*8($16)

	mf_fpcr $f0
	lda	$1, 8
	stt	$f0, UC_SIGCTX+SC_FPCR($16)

	/* The return address of getcontext is the restart pc.  */
	stq	$26, UC_SIGCTX+SC_PC($16)

	/* Userlevel always has a processor status word of 8.  */
	stq	$1, UC_SIGCTX+SC_PS($16)

	/* Save registers around the syscall.  We preserve $17
	   for the benefit of swapcontext.  */
	subq	$30, 4*8, $30
	cfi_adjust_cfa_offset(4*8)
	stq	$0, 0($30)
	cfi_rel_offset(64, 0)
	stq	$16, 8($30)
	stq	$17, 16($30)

	/* Save the current signal mask.  Whee, there are three
	   copies of this in the alpha ucontext_t.  */
	lda	$16, SIG_BLOCK
	lda	$17, 0
	lda	$0, __NR_osf_sigprocmask
	callsys

	ldq	$16, 8($30)
	ldq	$17, 16($30)

	stq	$0, UC_OSF_SIGMASK($16)
	stq	$0, UC_SIGCTX+SC_MASK($16)
	stq	$0, UC_SIGMASK($16)
	stq	$31, UC_SIGMASK + 1*8($16)
	stq	$31, UC_SIGMASK + 2*8($16)
	stq	$31, UC_SIGMASK + 3*8($16)
	stq	$31, UC_SIGMASK + 4*8($16)
	stq	$31, UC_SIGMASK + 5*8($16)
	stq	$31, UC_SIGMASK + 6*8($16)
	stq	$31, UC_SIGMASK + 7*8($16)
	stq	$31, UC_SIGMASK + 8*8($16)
	stq	$31, UC_SIGMASK + 9*8($16)
	stq	$31, UC_SIGMASK +10*8($16)
	stq	$31, UC_SIGMASK +11*8($16)
	stq	$31, UC_SIGMASK +12*8($16)
	stq	$31, UC_SIGMASK +13*8($16)
	stq	$31, UC_SIGMASK +14*8($16)
	stq	$31, UC_SIGMASK +15*8($16)

	ldq	$0, 0($30)
	addq	$30, 4*8, $30
	cfi_register (64, 0)
	cfi_adjust_cfa_offset(-4*8)
	ret	$31, ($0), 1

	cfi_endproc
	.size	__getcontext_x, .-__getcontext_x
	.type	__getcontext_x, @function
