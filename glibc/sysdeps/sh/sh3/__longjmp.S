/* longjmp for SH.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>
#define _SETJMP_H
#define _ASM
#include <bits/setjmp.h>

/* __longjmp(jmpbuf, val) */

ENTRY (__longjmp)
	mov.l	@r4+, r8
	mov.l	@r4+, r9
	mov.l	@r4+, r10
	mov.l	@r4+, r11
	mov.l	@r4+, r12
	mov	r5, r0		/* get the return value in place */
	tst	r0, r0
	bf.s	1f
	 mov.l	@r4+, r13
	mov	#1,r0		/* can't let setjmp() return zero! */
1:
#ifdef PTR_DEMANGLE
	mov.l	@r4+, r2
	PTR_DEMANGLE (r2, r1)
	mov	r2, r14
	mov.l	@r4+, r2
	PTR_DEMANGLE2 (r2, r1)
# ifdef CHECK_SP
	CHECK_SP (r2)
# endif
	mov	r2, r15
	mov.l	@r4+, r2
	PTR_DEMANGLE2 (r2, r1)
	lds	r2, pr
	mov	#0, r1
#else
	mov.l	@r4+, r14
# ifdef CHECK_SP
	mov.l	@r4, r2
	CHECK_SP (r2)
# endif
	mov.l	@r4+, r15
	lds.l	@r4+, pr
#endif
	rts
	 ldc.l	@r4+, gbr
END (__longjmp)
