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


ENTRY(__longjmp)
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

	mov     a1, v0
	ldq     s0, JB_S0*8(a0)
	ldq     s1, JB_S1*8(a0)
	ldq     s2, JB_S2*8(a0)
	ldq     s3, JB_S3*8(a0)
	ldq     s4, JB_S4*8(a0)
	ldq     s5, JB_S5*8(a0)
	ldq     ra, JB_PC*8(a0)
	ldq     fp, JB_FP*8(a0)
	ldq     t0, JB_SP*8(a0)
	ldt     $f2, JB_F2*8(a0)
	ldt     $f3, JB_F3*8(a0)
	ldt     $f4, JB_F4*8(a0)
	ldt     $f5, JB_F5*8(a0)
	ldt     $f6, JB_F6*8(a0)
	ldt     $f7, JB_F7*8(a0)
	ldt     $f8, JB_F8*8(a0)
	ldt     $f9, JB_F9*8(a0)
#ifdef PTR_DEMANGLE
	PTR_DEMANGLE(ra, t1)
	PTR_DEMANGLE2(t0, t1)
	PTR_DEMANGLE2(fp, t1)
#endif
	cmoveq  v0, 1, v0
	mov     t0, sp
	ret

END(__longjmp)
