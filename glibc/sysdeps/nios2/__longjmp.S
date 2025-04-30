/* longjmp for Nios II.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>
#include <jmpbuf-offsets.h>

/* __longjmp(jmpbuf, val) */

ENTRY (__longjmp)
	mov	r2, r5
	bne	r5, zero, 1f
	movi	r2, 1
1:
#ifdef	PTR_DEMANGLE
	PTR_MANGLE_GUARD (r3)
	ldw	r5,  (JB_RA*4)(r4)
	ldw	r6,  (JB_SP*4)(r4)
	PTR_DEMANGLE (r5, r5, r3)
	PTR_DEMANGLE (r6, r6, r3)
#endif

	cfi_def_cfa (r4, 0)
	cfi_offset (r16, JB_R16*4)
	cfi_offset (r17, JB_R17*4)
	cfi_offset (r18, JB_R18*4)
	cfi_offset (r19, JB_R19*4)
	cfi_offset (r20, JB_R20*4)
	cfi_offset (r21, JB_R21*4)
	cfi_offset (r22, JB_R22*4)
	cfi_offset (fp, JB_FP*4)
#ifdef PTR_DEMANGLE
	cfi_register (ra, r5)
	cfi_register (sp, r7)
#else
	cfi_offset (ra, JB_RA*4)
	cfi_offset (sp, JB_SP*4)
#endif
	ldw	r16, (JB_R16*4)(r4)
	ldw	r17, (JB_R17*4)(r4)
	ldw	r18, (JB_R18*4)(r4)
	ldw	r19, (JB_R19*4)(r4)
	ldw	r20, (JB_R20*4)(r4)
	ldw	r21, (JB_R21*4)(r4)
	ldw	r22, (JB_R22*4)(r4)
	ldw	fp,  (JB_FP*4)(r4)
#ifdef PTR_DEMANGLE
	mov	ra, r5
	mov	sp, r6
#else
	ldw	ra,  (JB_RA*4)(r4)
	ldw	sp,  (JB_SP*4)(r4)
#endif
	cfi_same_value (r16)
	cfi_same_value (r17)
	cfi_same_value (r18)
	cfi_same_value (r19)
	cfi_same_value (r20)
	cfi_same_value (r21)
	cfi_same_value (r22)
	cfi_same_value (fp)
	cfi_same_value (ra)
	cfi_same_value (sp)

	ret
END (__longjmp)
