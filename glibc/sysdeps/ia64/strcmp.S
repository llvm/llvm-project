/* Optimized version of the standard strcmp() function.
   This file is part of the GNU C Library.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   Contributed by Dan Pop <Dan.Pop@cern.ch>.

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

/* Return: the result of the comparison

   Inputs:
        in0:    s1
        in1:    s2

   Unlike memcmp(), this function is optimized for mismatches within the
   first few characters.  */

#include <sysdep.h>
#undef ret

#define s1		in0
#define s2		in1

#define val1		r15
#define val2		r16


ENTRY(strcmp)
	alloc	r2 = ar.pfs, 2, 0, 0, 0
.loop:
	ld1	val1 = [s1], 1
	ld1	val2 = [s2], 1
	cmp.eq	p6, p0 = r0, r0		// set p6
	;;
	cmp.ne.and p6, p0 = val1, r0
	cmp.ne.and p6, p0 = val2, r0
	cmp.eq.and p6, p0 = val1, val2
(p6)	br.cond.sptk .loop
	sub	ret0 = val1, val2
	br.ret.sptk.many b0
END(strcmp)
libc_hidden_builtin_def (strcmp)
