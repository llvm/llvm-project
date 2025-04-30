/* setjmp for Nios II.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

	.text
ENTRY(setjmp)
	movi	r5, 1
	br	__sigsetjmp
END(setjmp)

ENTRY(_setjmp)
	mov	r5, zero
	br	__sigsetjmp
END(_setjmp)
libc_hidden_def (_setjmp)

/* Save the current program position in ENV and return 0.  */
ENTRY(__sigsetjmp)
#ifdef	PTR_MANGLE
	PTR_MANGLE_GUARD (r9)
#endif
	stw	r16, (JB_R16*4)(r4)
	stw	r17, (JB_R17*4)(r4)
	stw	r18, (JB_R18*4)(r4)
	stw	r19, (JB_R19*4)(r4)
	stw	r20, (JB_R20*4)(r4)
	stw	r21, (JB_R21*4)(r4)
	stw	r22, (JB_R22*4)(r4)
	stw	fp,  (JB_FP*4)(r4)
#ifdef	PTR_MANGLE
	PTR_MANGLE (r6, ra, r9)
	PTR_MANGLE (r7, sp, r9)
	stw	r6,  (JB_RA*4)(r4)
	stw	r7,  (JB_SP*4)(r4)
#else
	stw	ra,  (JB_RA*4)(r4)
	stw	sp,  (JB_SP*4)(r4)
#endif
#if IS_IN (rtld)
	/* In ld.so we never save the signal mask.  */
	mov r2, zero
	ret
#else
	/* Save the signal mask if requested.  */
#if defined(__PIC__) || defined(PIC)
        nextpc r2
1:	movhi r3, %hiadj(__sigjmp_save - 1b)
	addi r3, r3, %lo(__sigjmp_save - 1b)
	add r2, r2, r3
	jmp r2
#else
	jmpi __sigjmp_save
#endif
#endif
END (__sigsetjmp)
libc_hidden_def (__sigsetjmp)
