/* PowerPC-specific implementation of profiling support.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

/* This would be bad.  */
#ifdef PROF
#undef PROF
#endif

#include <sysdep.h>

/* We do profiling as described in the SYSV ELF ABI, except that glibc
   _mcount manages its own counters.  The caller has put the address the
   caller will return to in the usual place on the stack, 4(r1).  _mcount
   is responsible for ensuring that when it returns no argument-passing
   registers are disturbed, and that the LR is set back to (what the
   caller sees as) 4(r1).

   This is intended so that the following code can be inserted at the
   front of any routine without changing the routine:

	.data
	mflr	r0
	stw	r0,4(r1)
	bl	_mcount
*/

ENTRY(_mcount)
#if defined PIC && !defined SHARED
# define CALLER_LR_OFFSET 68
	stwu	r1,-64(r1)
	cfi_adjust_cfa_offset (64)
	stw	r30, 48(r1)
	cfi_rel_offset (r30, 48)
#else
# define CALLER_LR_OFFSET 52
	stwu	r1,-48(r1)
	cfi_adjust_cfa_offset (48)
#endif
/* We need to save the parameter-passing registers.  */
	stw	r3, 12(r1)
	stw	r4, 16(r1)
	stw	r5, 20(r1)
	stw	r6, 24(r1)
	mflr	r4
#if defined PIC && !defined SHARED
	bcl	20,31,0f
0:
	mflr	r30
	addis	r30, r30, _GLOBAL_OFFSET_TABLE_-0b@ha
	addi	r30, r30, _GLOBAL_OFFSET_TABLE_-0b@l
#endif
	lwz	r3, CALLER_LR_OFFSET(r1)
	mfcr	r5
	stw	r7, 28(r1)
	stw	r8, 32(r1)
	stw	r9, 36(r1)
	stw	r10,40(r1)
	stw	r4, 44(r1)
	cfi_rel_offset (lr, 44)
	stw	r5,  8(r1)
#ifndef SHARED
	bl	JUMPTARGET(__mcount_internal)
#else
	bl	__mcount_internal@local
#endif
 /* Restore the registers...  */
	lwz     r6,  8(r1)
	lwz	r0, 44(r1)
	lwz	r3, 12(r1)
	mtctr	r0
	lwz	r4, 16(r1)
	mtcrf	0xff,r6
	lwz	r5, 20(r1)
	lwz	r6, 24(r1)
	lwz	r0, CALLER_LR_OFFSET(r1)
	lwz	r7, 28(r1)
	lwz	r8, 32(r1)
	mtlr	r0
	lwz	r9, 36(r1)
	lwz	r10,40(r1)
 /* ...unwind the stack frame, and return to your usual programming.  */
#if defined PIC && !defined SHARED
	lwz	r30, 48(r1)
	addi	r1,r1,64
#else
	addi	r1,r1,48
#endif
	bctr
END(_mcount)
