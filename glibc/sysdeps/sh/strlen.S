/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Kazumoto Kojima <kkojima@rr.iij4u.or.jp>

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

/* size_t strlen (const char *s)  */

ENTRY(strlen)
	mov	r4, r0
	and	#3, r0
	tst	r0, r0
	bt/s	1f
	mov	#0, r2

	add	#-1, r0
	shll2	r0
	shll	r0
	braf	r0
	nop

	mov.b	@r4+, r1
	tst	r1, r1
	bt	8f
	add	#1, r2

	mov.b	@r4+, r1
	tst	r1, r1
	bt	8f
	add	#1, r2

	mov.b	@r4+, r1
	tst	r1, r1
	bt	8f
	add	#1, r2

1:
	mov	#0, r3
2:
	mov.l	@r4+, r1
	cmp/str	r3, r1
	bf/s	2b
	add	#4, r2

	add	#-4, r2
#ifdef __BIG_ENDIAN__
	swap.b	r1, r1
	swap.w	r1, r1
	swap.b	r1, r1
#endif
	extu.b	r1, r0
	tst	r0, r0
	bt/s	8f
	shlr8	r1
	add	#1, r2
	extu.b	r1, r0
	tst	r0, r0
	bt/s	8f
	shlr8	r1
	add	#1, r2
	extu.b	r1, r0
	tst	r0, r0
	bt	8f
	add	#1, r2
8:
	rts
	mov	r2, r0
END(strlen)
libc_hidden_builtin_def (strlen)
