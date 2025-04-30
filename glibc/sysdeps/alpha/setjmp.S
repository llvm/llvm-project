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

#define __ASSEMBLY__

#include <sysdep.h>
#include <jmpbuf-offsets.h>

	.ent __sigsetjmp
	.global __sigsetjmp
__sigsetjmp:
	ldgp    gp, 0(pv)

$sigsetjmp_local:
#ifndef PIC
#define FRAME	16
	subq    sp, FRAME, sp
	.frame  sp, FRAME, ra, 0
	stq     ra, 0(sp)
	.mask   0x04000000, -FRAME
#else
#define FRAME	0
	.frame	sp, FRAME, ra, 0
#endif
#ifdef PROF
	.set noat
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.set at
#endif
	.prologue 1

	stq	s0, JB_S0*8(a0)
	stq	s1, JB_S1*8(a0)
	stq	s2, JB_S2*8(a0)
	stq	s3, JB_S3*8(a0)
	stq	s4, JB_S4*8(a0)
	stq	s5, JB_S5*8(a0)
#ifdef PTR_MANGLE
	PTR_MANGLE(t1, ra, t0)
	stq	t1, JB_PC*8(a0)
#else
	stq	ra, JB_PC*8(a0)
#endif
#if defined(PTR_MANGLE) && FRAME == 0
	PTR_MANGLE2(t1, sp, t0)
#else
	addq	sp, FRAME, t1
# ifdef PTR_MANGLE
	PTR_MANGLE2(t1, t1, t0)
# endif
#endif
	stq	t1, JB_SP*8(a0)
#ifdef PTR_MANGLE
	PTR_MANGLE2(t1, fp, t0)
	stq	t1, JB_FP*8(a0)
#else
	stq	fp, JB_FP*8(a0)
#endif
	stt	$f2, JB_F2*8(a0)
	stt	$f3, JB_F3*8(a0)
	stt	$f4, JB_F4*8(a0)
	stt	$f5, JB_F5*8(a0)
	stt	$f6, JB_F6*8(a0)
	stt	$f7, JB_F7*8(a0)
	stt	$f8, JB_F8*8(a0)
	stt	$f9, JB_F9*8(a0)

#ifndef PIC
	/* Call to C to (potentially) save our signal mask.  */
	jsr	ra, __sigjmp_save
	ldq	ra, 0(sp)
	addq	sp, 16, sp
	ret
#elif IS_IN (rtld)
	/* In ld.so we never save the signal mask.  */
	mov	0, v0
	ret
#else
	/* Tailcall to save the signal mask.  */
	br	$31, __sigjmp_save	!samegp
#endif

END(__sigsetjmp)
hidden_def (__sigsetjmp)

/* Put these traditional entry points in the same file so that we can
   elide much of the nonsense in trying to jmp to the real function.  */

ENTRY(_setjmp)
	ldgp	gp, 0(pv)
	.prologue 1
	mov	0, a1
	br	$sigsetjmp_local
END(_setjmp)
libc_hidden_def (_setjmp)

ENTRY(setjmp)
	ldgp	gp, 0(pv)
	.prologue 1
	mov	1, a1
	br	$sigsetjmp_local
END(setjmp)

weak_extern(_setjmp)
weak_extern(setjmp)
