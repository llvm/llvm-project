/* longjmp for ARC.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

/* @ r0 = jump buffer from which regs will be restored
   @ r1 = value that setjmp( ) will return due to this longjmp.  */

ENTRY (__longjmp)

	LDR (blink, r0,  0)
	LDR (sp,    r0,  1)
	LDR (fp,    r0,  2)
	LDR (gp,    r0,  3)

	LDR (r13,   r0,  4)
	LDR (r14,   r0,  5)
	LDR (r15,   r0,  6)
	LDR (r16,   r0,  7)
	LDR (r17,   r0,  8)
	LDR (r18,   r0,  9)
	LDR (r19,   r0, 10)
	LDR (r20,   r0, 11)
	LDR (r21,   r0, 12)
	LDR (r22,   r0, 13)
	LDR (r23,   r0, 14)
	LDR (r24,   r0, 15)

	mov.f  r0, r1
	j.d    [blink]
	mov.z  r0, 1	/* don't return 0 to setjmp callsite from longjmp.  */

END (__longjmp)
