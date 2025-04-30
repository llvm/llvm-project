/* setjmp for HPPA.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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
#define _ASM
#define _SETJMP_H
#include <bits/setjmp.h>


	.text
	.align 4
	.import __sigjmp_save, code
	.globl __sigsetjmp
	.export __sigsetjmp, code
	.proc
	.callinfo
__sigsetjmp:
	stw	%r3, 0(%r26)
	/* This padding exists for unknown historical reasons.  */
	/* 4 - 4 bytes of padding.  */
	stw	%r4, 8(%r26)
	stw	%r5, 12(%r26)
	stw	%r6, 16(%r26)
	stw	%r7, 20(%r26)
	stw	%r8, 24(%r26)
	stw	%r9, 28(%r26)
	stw	%r10, 32(%r26)
	stw	%r11, 36(%r26)
	stw	%r12, 40(%r26)
	stw	%r13, 44(%r26)
	stw	%r14, 48(%r26)
	stw	%r15, 52(%r26)
	stw	%r16, 56(%r26)
	stw	%r17, 60(%r26)
	stw	%r18, 64(%r26)
	stw	%r19, 68(%r26)
	stw	%r27, 72(%r26)
	stw	%r30, 76(%r26)

	stw	%rp, 80(%r26)
	/* This padding exists to ensure double alignment for fprs.  */
	/* 84 - 4 bytes of padding.  */
	ldo	88(%r26),%r1
	fstds,ma %fr12, 8(%r1) /* 88 */
	fstds,ma %fr13, 8(%r1) /* 96 */
	fstds,ma %fr14, 8(%r1) /* 104 */
	fstds,ma %fr15, 8(%r1) /* 112 */
	fstds,ma %fr16, 8(%r1) /* 120 */
	fstds,ma %fr17, 8(%r1) /* 128 */
	fstds,ma %fr18, 8(%r1) /* 136 */
	fstds,ma %fr19, 8(%r1) /* 144 */
	fstds,ma %fr20, 8(%r1) /* 152 */
	fstds	 %fr21, 0(%r1) /* 160 */
	/* Total of 168 bytes.  */
	b __sigjmp_save
	nop
	.procend
hidden_def (__sigsetjmp)
