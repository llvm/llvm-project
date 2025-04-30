/* Optimized version of the standard strcpy() function.
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

/* Return: dest

   Inputs:
        in0:    dest
        in1:    src

   In this form, it assumes little endian mode.  For big endian mode,
   the two shifts in .l2 must be inverted:

	shl   	value = r[1], sh1   	// value = w0 << sh1
	shr.u   tmp = r[0], sh2   	// tmp = w1 >> sh2
 */

#include <sysdep.h>
#undef ret

#define saved_lc	r15
#define saved_pr	r16
#define thresh		r17
#define dest		r19
#define src		r20
#define len		r21
#define asrc		r22
#define tmp		r23
#define pos		r24
#define w0		r25
#define w1		r26
#define c		r27
#define sh2		r28
#define	sh1		r29
#define loopcnt		r30
#define	value		r31

ENTRY(strcpy)
	.prologue
	alloc 	r2 = ar.pfs, 2, 0, 30, 32

#define MEMLAT 2
	.rotr	r[MEMLAT + 2]
	.rotp	p[MEMLAT + 1]

	mov	ret0 = in0		// return value = dest
	.save pr, saved_pr
	mov	saved_pr = pr           // save the predicate registers
	.save ar.lc, saved_lc
        mov 	saved_lc = ar.lc	// save the loop counter
	.body
	sub	tmp = r0, in0 ;;	// tmp = -dest
	mov 	dest = in0		// dest
	mov 	src = in1		// src
	and	loopcnt = 7, tmp ;;	// loopcnt = -dest % 8
	cmp.eq	p6, p0 = loopcnt, r0
	adds	loopcnt = -1, loopcnt	// --loopcnt
(p6)	br.cond.sptk .dest_aligned ;;
	mov	ar.lc = loopcnt
.l1:					// copy -dest % 8 bytes
	ld1	c = [src], 1		// c = *src++
	;;
	st1	[dest] = c, 1		// *dest++ = c
	cmp.eq	p6, p0 = c, r0
(p6)	br.cond.dpnt .restore_and_exit
	br.cloop.dptk .l1 ;;
.dest_aligned:
	and	sh1 = 7, src 		// sh1 = src % 8
	mov	ar.lc = -1		// "infinite" loop
	and	asrc = -8, src ;;	// asrc = src & -OPSIZ  -- align src
	sub	thresh = 8, sh1
	mov	pr.rot = 1 << 16	// set rotating predicates
	cmp.ne	p7, p0 = r0, r0		// clear p7
	shl	sh1 = sh1, 3 ;;		// sh1 = 8 * (src % 8)
	sub	sh2 = 64, sh1		// sh2 = 64 - sh1
	cmp.eq  p6, p0 = sh1, r0 	// is the src aligned?
(p6)    br.cond.sptk .src_aligned ;;
	ld8	r[1] = [asrc],8 ;;

	.align	32
.l2:
	ld8.s	r[0] = [asrc], 8
	shr.u	value = r[1], sh1 ;; 	// value = w0 >> sh1
	czx1.r	pos = value ;;		// do we have an "early" zero
	cmp.lt	p7, p0 = pos, thresh	// in w0 >> sh1?
(p7)	br.cond.dpnt .found0
	chk.s	r[0], .recovery2	// it is safe to do that only
.back2:					// after the previous test
	shl	tmp = r[0], sh2  	// tmp = w1 << sh2
	;;
	or	value = value, tmp ;;	// value |= tmp
	czx1.r	pos = value ;;
	cmp.ne	p7, p0 = 8, pos
(p7)	br.cond.dpnt .found0
	st8	[dest] = value, 8	// store val to dest
	br.ctop.dptk    .l2 ;;
.src_aligned:
.l3:
(p[0])		ld8.s	r[0] = [src], 8
(p[MEMLAT])	chk.s	r[MEMLAT], .recovery3
.back3:
(p[MEMLAT])	mov	value = r[MEMLAT]
(p[MEMLAT])	czx1.r	pos = r[MEMLAT] ;;
(p[MEMLAT])	cmp.ne	p7, p0 = 8, pos
(p7)		br.cond.dpnt .found0
(p[MEMLAT])	st8	[dest] = r[MEMLAT], 8
		br.ctop.dptk .l3 ;;
.found0:
	mov	ar.lc = pos
.l4:
	extr.u	c = value, 0, 8		// c = value & 0xff
	shr.u	value = value, 8
	;;
	st1	[dest] = c, 1
	br.cloop.dptk	.l4 ;;
.restore_and_exit:
	mov 	ar.lc = saved_lc	// restore the loop counter
	mov	pr = saved_pr, -1	// restore the predicate registers
	br.ret.sptk.many b0
.recovery2:
	add	tmp = -8, asrc ;;
	ld8	r[0] = [tmp]
	br.cond.sptk .back2
.recovery3:
	add	tmp = -(MEMLAT + 1) * 8, src ;;
	ld8	r[MEMLAT] = [tmp]
	br.cond.sptk .back3
END(strcpy)
libc_hidden_builtin_def (strcpy)
