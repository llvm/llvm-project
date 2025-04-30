/* setjmp for ARC.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

/* Upon entry r0 = jump buffer into which regs will be saved.  */
ENTRY (setjmp)
	b.d	__sigsetjmp
	mov	r1, 1		/* save signals.  */
END (setjmp)

/* Upon entry r0 = jump buffer into which regs will be saved.  */
ENTRY (_setjmp)
	b.d	__sigsetjmp
	mov	r1, 0		/* don't save signals.  */
END (_setjmp)
libc_hidden_def (_setjmp)

/* Upon entry
   r0 = jump buffer into which regs will be saved
   r1 = do we need to save signals.  */
ENTRY (__sigsetjmp)

	/* Make a note of where longjmp will return to.
	   that will be right next to this setjmp call-site in BLINK
	   since "C" caller of this routine will do a branch-n-link.  */

	STR (blink, r0,  0)
	STR (sp,    r0,  1)
	STR (fp,    r0,  2)
	STR (gp,    r0,  3)

	STR (r13,   r0,  4)
	STR (r14,   r0,  5)
	STR (r15,   r0,  6)
	STR (r16,   r0,  7)
	STR (r17,   r0,  8)
	STR (r18,   r0,  9)
	STR (r19,   r0, 10)
	STR (r20,   r0, 11)
	STR (r21,   r0, 12)
	STR (r22,   r0, 13)
	STR (r23,   r0, 14)
	STR (r24,   r0, 15)

	b    __sigjmp_save

END (__sigsetjmp)

libc_hidden_def (__sigsetjmp)
