/* Optimized version of the standard strchr() function.
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

/* Return: the address of the first occurence of chr in str or NULL

   Inputs:
        in0:    str
        in1:    chr

   A modified version of memchr.S, the search ends when the character is
   found or the terminating null character is encountered.

   This implementation assumes little endian mode.  For big endian mode,
   the instruction czx1.r should be replaced by czx1.l.  */

#include <sysdep.h>
#undef ret

#define saved_lc	r18
#define poschr		r19
#define pos0		r20
#define val1		r21
#define val2		r22
#define tmp		r24
#define chrx8		r25
#define loopcnt		r30

#define str		in0
#define chr		in1

ENTRY(strchr)
	.prologue
	alloc r2 = ar.pfs, 2, 0, 0, 0
	.save ar.lc, saved_lc
        mov 	saved_lc = ar.lc 	// save the loop counter
	.body
	mov 	ret0 = str
	and 	tmp = 7, str		// tmp = str % 8
	mux1	chrx8 = chr, @brcst
	extr.u	chr = chr, 0, 8		// retain only the last byte
	cmp.ne	p8, p0 = r0, r0		// clear p8
	;;
	sub	loopcnt = 8, tmp	// loopcnt = 8 - tmp
	cmp.eq	p6, p0 = tmp, r0
(p6)	br.cond.sptk	.str_aligned;;
	adds	loopcnt = -1, loopcnt;;
	mov	ar.lc = loopcnt
.l1:
	ld1	val2 = [ret0], 1
	;;
	cmp.eq	p6, p0 = val2, chr
	cmp.eq	p7, p0 = val2, r0
(p6)	br.cond.spnt	.restore_and_exit
(p7)	br.cond.spnt	.notfound
	br.cloop.sptk	.l1
.str_aligned:
	ld8	val1 = [ret0], 8;;
	nop.b	0
	nop.b 	0
.l2:
	ld8.s	val2 = [ret0], 8	// don't bomb out here
	czx1.r	pos0 = val1
	xor	tmp = val1, chrx8	// if val1 contains chr, tmp will
	;;				// contain a zero in its position
	czx1.r	poschr = tmp
	cmp.ne	p6, p0 = 8, pos0
	;;
	cmp.ne	p7, p0 = 8, poschr
(p7)	br.cond.spnt .foundit
(p6)	br.cond.spnt .notfound
	chk.s	val2, .recovery
.back:
	mov	val1 = val2
	br.cond.dptk .l2
.foundit:
(p6)	cmp.lt	p8, p0 = pos0, poschr	// we found chr and null in the word
(p8)	br.cond.spnt .notfound		// null was found before chr
	add	ret0 = ret0, poschr ;;
	adds	ret0 = -15, ret0 ;;	// should be -16, but we decrement
.restore_and_exit:			// ret0 in the next instruction
	adds	ret0 = -1, ret0		// ret0 was pointing 1 char too far
	mov 	ar.lc = saved_lc	// restore the loop counter
	br.ret.sptk.many b0
.notfound:
	mov	ret0 = r0		// return NULL if null was found
	mov 	ar.lc = saved_lc
	br.ret.sptk.many b0
.recovery:
	adds	ret0 = -8, ret0;;
	ld8	val2 = [ret0], 8	// bomb out here
	br.cond.sptk	.back
END(strchr)

weak_alias (strchr, index)
libc_hidden_builtin_def (strchr)
