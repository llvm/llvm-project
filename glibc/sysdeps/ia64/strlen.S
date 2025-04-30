/* Optimized version of the standard strlen() function.
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

/* Return: the length of the input string

   Input:
        in0:    str

   Look for the null character byte by byte, until we reach a word aligned
   address, then search word by word, using the czx instruction.  We're
   also doing one word of read ahead, which could cause problems if the
   null character is on the last word of a page and the next page is not
   mapped in the process address space.  Hence the use of the speculative
   load.

   This implementation assumes little endian mode.  For big endian mode,
   the instruction czx1.r should be replaced by czx1.l.  */

#include <sysdep.h>
#undef ret

#define saved_lc	r18
#define str		r19
#define pos0		r20
#define val1		r21
#define val2		r22
#define origadd		r23
#define tmp		r24
#define loopcnt		r30
#define len		ret0

ENTRY(strlen)
	.prologue
	alloc r2 = ar.pfs, 1, 0, 0, 0
	.save ar.lc, saved_lc
        mov 	saved_lc = ar.lc 	// save the loop counter
	.body
	mov 	str = in0
	mov 	len = r0		// len = 0
	and 	tmp = 7, in0		// tmp = str % 8
	;;
	sub	loopcnt = 8, tmp	// loopcnt = 8 - tmp
	cmp.eq	p6, p0 = tmp, r0
(p6)	br.cond.sptk	.str_aligned;;
	adds	loopcnt = -1, loopcnt;;
	mov	ar.lc = loopcnt
.l1:
	ld1	val2 = [str], 1
	;;
	cmp.eq	p6, p0 = val2, r0
(p6)	br.cond.spnt	.restore_and_exit
	adds	len = 1, len
	br.cloop.dptk	.l1
.str_aligned:
	mov	origadd = str		// origadd = orig
	ld8	val1 = [str], 8;;
	nop.b	0
	nop.b 	0
.l2:	ld8.s	val2 = [str], 8		// don't bomb out here
	czx1.r	pos0 = val1
	;;
	cmp.ne	p6, p0 = 8, pos0
(p6)	br.cond.spnt .foundit
	chk.s	val2, .recovery
.back:
	mov	val1 = val2
	br.cond.dptk	.l2
.foundit:
	sub	tmp = str, origadd	// tmp = crt address - orig
	add	len = len, pos0;;
	add	len = len, tmp;;
	adds	len = -16, len
.restore_and_exit:
	mov ar.lc = saved_lc		// restore the loop counter
	br.ret.sptk.many b0
.recovery:
	adds	str = -8, str;;
	ld8	val2 = [str], 8		// bomb out here
	br.cond.sptk	.back
END(strlen)
libc_hidden_builtin_def (strlen)
