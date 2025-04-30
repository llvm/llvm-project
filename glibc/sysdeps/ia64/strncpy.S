/* Optimized version of the standard strncpy() function.
   This file is part of the GNU C Library.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   Contributed by Dan Pop <Dan.Pop@cern.ch>
	      and Jakub Jelinek <jakub@redhat.com>.

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
	in2:	len

   In this form, it assumes little endian mode.
 */

#include <sysdep.h>
#undef ret

#define saved_lc	r15
#define saved_pr	r16
#define thresh		r17
#define dest		r18
#define dest2		r19
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

ENTRY(strncpy)
	.prologue
	alloc 	r2 = ar.pfs, 3, 0, 29, 32

#define MEMLAT 2
	.rotr	r[MEMLAT + 2]
	.rotp	p[MEMLAT + 1]

	mov	ret0 = in0		// return value = dest
	.save pr, saved_pr
	mov	saved_pr = pr           // save the predicate registers
	.save ar.lc, saved_lc
	mov 	saved_lc = ar.lc	// save the loop counter
	mov	ar.ec = 0		// ec is not guaranteed to
					// be zero upon function entry
	.body
	cmp.geu p6, p5 = 24, in2
(p6)	br.cond.spnt .short_len
	sub	tmp = r0, in0 ;;	// tmp = -dest
	mov	len = in2		// len
	mov 	dest = in0		// dest
	mov 	src = in1		// src
	and	tmp = 7, tmp ;;		// loopcnt = -dest % 8
	cmp.eq	p6, p7 = tmp, r0
	adds	loopcnt = -1, tmp	// --loopcnt
(p6)	br.cond.sptk .dest_aligned ;;
	sub	len = len, tmp		// len -= -dest % 8
	mov	ar.lc = loopcnt
.l1:					// copy -dest % 8 bytes
(p5)	ld1	c = [src], 1		// c = *src++
	;;
	st1	[dest] = c, 1		// *dest++ = c
	cmp.ne	p5, p7 = c, r0
	br.cloop.dptk .l1 ;;
(p7)	br.cond.dpnt	.found0_align

.dest_aligned:				// p7 should be cleared here
	shr.u	c = len, 3		// c = len / 8
	and	sh1 = 7, src 		// sh1 = src % 8
	and	asrc = -8, src ;;	// asrc = src & -OPSIZ  -- align src
	adds	c = (MEMLAT-1), c	// c = (len / 8) + MEMLAT - 1
	sub	thresh = 8, sh1
	mov	pr.rot = 1 << 16	// set rotating predicates
	shl	sh1 = sh1, 3 ;;		// sh1 = 8 * (src % 8)
	mov	ar.lc = c		// "infinite" loop
	sub	sh2 = 64, sh1		// sh2 = 64 - sh1
	cmp.eq  p6, p0 = sh1, r0 	// is the src aligned?
(p6)    br.cond.sptk .src_aligned
	adds	c = -(MEMLAT-1), c ;;	// c = (len / 8)
	ld8	r[1] = [asrc],8
	mov	ar.lc = c ;;

	.align	32
.l2:
(p6)	st8	[dest] = value, 8	// store val to dest
	ld8.s	r[0] = [asrc], 8
	shr.u	value = r[1], sh1 ;; 	// value = w0 >> sh1
	czx1.r	pos = value ;;		// do we have an "early" zero
	cmp.lt	p7, p0 = pos, thresh	// in w0 >> sh1?
	adds	len = -8, len		// len -= 8
(p7)	br.cond.dpnt .nonalign_found0
	chk.s	r[0], .recovery2	// it is safe to do that only
.back2:					// after the previous test
	shl	tmp = r[0], sh2  	// tmp = w1 << sh2
	;;
	or	value = value, tmp ;;	// value |= tmp
	czx1.r	pos = value ;;
	cmp.ne	p7, p6 = 8, pos
(p7)	br.cond.dpnt .nonalign_found0
	br.ctop.dptk    .l2 ;;
	adds	len = 8, len
	br.cond.sptk	.not_found0 ;;
.nonalign_found0:
	cmp.gtu	p6, p0 = -8, len
(p6)	br.cond.dptk .found0
	adds	len = 8, len
	br.cond.sptk	.not_found0 ;;

	.align	32
.src_aligned:
.l3:
(p[0])		ld8.s	r[0] = [src], 8
(p[MEMLAT])	chk.s	r[MEMLAT], .recovery3
.back3:
(p[MEMLAT])	mov	value = r[MEMLAT]
(p[MEMLAT])	czx1.r	pos = r[MEMLAT] ;;
(p[MEMLAT])	cmp.ne	p7, p0 = 8, pos
(p[MEMLAT])	adds	len = -8, len	// len -= 8
(p7)		br.cond.dpnt .found0
(p[MEMLAT])	st8	[dest] = r[MEMLAT], 8
		br.ctop.dptk .l3 ;;

	chk.s	r[MEMLAT-1], .recovery4
.back4:
	mov	value = r[MEMLAT-1]

.not_found0:
	cmp.eq	p5, p6 = len, r0
	adds	len = -1, len
(p5)	br.cond.dptk	.restore_and_exit ;;
	mov	ar.lc = len
.l4:
(p6)	extr.u	c = value, 0, 8		// c = value & 0xff
(p6)	shr.u	value = value, 8 ;;
	st1	[dest] = c, 1
	cmp.ne	p6, p0 = c, r0
	br.cloop.dptk	.l4
	br.cond.sptk	.restore_and_exit

.found0_align:
	mov	pos = 0
	adds	len = -8, len
	mov	value = 0 ;;
.found0:
	shl	tmp = pos, 3
	shr.u	loopcnt = len, 4	// loopcnt = len / 16
	mov	c = -1 ;;
	cmp.eq	p6, p0 = loopcnt, r0
	adds	loopcnt = -1, loopcnt
	shl	c = c, tmp ;;
	and	len = 0xf, len
	andcm	value = value, c
	mov	ar.lc = loopcnt ;;
	cmp.le	p7, p0 = 8, len
	adds	dest2 = 16, dest
	st8	[dest] = value, 8
	and	len = 0x7, len
(p6)	br.cond.dpnt	.l6 ;;
.l5:
	st8	[dest] = r0, 16
	st8	[dest2] = r0, 16
	br.cloop.dptk	.l5 ;;
.l6:
(p7)	st8	[dest] = r0, 8
	cmp.eq	p5, p0 = len, r0
	adds	len = -1, len
(p5)	br.cond.dptk .restore_and_exit ;;
	mov	ar.lc = len ;;
.l7:
	st1	[dest] = r0, 1
	br.cloop.dptk	.l7 ;;
.restore_and_exit:
	mov 	ar.lc = saved_lc	// restore the loop counter
	mov	pr = saved_pr, -1	// restore the predicate registers
	br.ret.sptk.many b0

.short_len:
	cmp.eq	p5, p0 = in2, r0
	adds	loopcnt = -1, in2
(p5)	br.cond.spnt .restore_and_exit ;;
	mov	ar.lc = loopcnt		// p6 should be set when we get here
.l8:
(p6)	ld1	c = [in1], 1		// c = *src++
	;;
	st1	[in0] = c, 1		// *dest++ = c
(p6)	cmp.ne	p6, p0 = c, r0
	br.cloop.dptk .l8
	;;
	mov 	ar.lc = saved_lc	// restore the loop counter
	mov	pr = saved_pr, -1	// restore the predicate registers
	br.ret.sptk.many b0
.recovery2:
	add	c = 8, len
	add	tmp = -8, asrc ;;
	cmp.gtu	p8, p5 = c, thresh ;;
(p8)	ld8	r[0] = [tmp]
(p5)	mov	r[0] = r0
	br.cond.sptk .back2
.recovery3:
	add	tmp = -(MEMLAT + 1) * 8, src ;;
	ld8	r[MEMLAT] = [tmp]
	br.cond.sptk .back3
.recovery4:
	cmp.eq	p5, p6 = len, r0
	add	tmp = -MEMLAT * 8, src ;;
(p6)	ld8	r[MEMLAT - 1] = [tmp]
(p5)	mov	r[MEMLAT - 1] = r0
	br.cond.sptk .back4
END(strncpy)
libc_hidden_builtin_def (strncpy)
