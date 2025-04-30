/* Optimized version of the standard memcmp() function.
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
        in0:    dest (aka s1)
        in1:    src  (aka s2)
        in2:    byte count

   In this form, it assumes little endian mode.  For big endian mode,
   the two shifts in .l2 must be inverted:

	shl   	tmp1[0] = r[1 + MEMLAT], sh1   // tmp1 = w0 << sh1
	shr.u   tmp2[0] = r[0 + MEMLAT], sh2   // tmp2 = w1 >> sh2

   and all the mux1 instructions should be replaced by plain mov's.  */

#include <sysdep.h>
#undef ret

#define OP_T_THRES 	16
#define OPSIZ 		8
#define MEMLAT		2

#define start		r15
#define saved_pr	r17
#define saved_lc	r18
#define dest		r19
#define src		r20
#define len		r21
#define asrc		r22
#define tmp		r23
#define value1		r24
#define value2		r25
#define sh2		r28
#define	sh1		r29
#define loopcnt		r30

ENTRY(memcmp)
	.prologue
	alloc 	r2 = ar.pfs, 3, 37, 0, 40

	.rotr	r[MEMLAT + 2], q[MEMLAT + 5], tmp1[4], tmp2[4], val[2]
	.rotp	p[MEMLAT + 4 + 1]

	mov	ret0 = r0		// by default return value = 0
	.save pr, saved_pr
	mov	saved_pr = pr		// save the predicate registers
	.save ar.lc, saved_lc
        mov 	saved_lc = ar.lc	// save the loop counter
	.body
	mov 	dest = in0		// dest
	mov 	src = in1		// src
	mov	len = in2		// len
	sub	tmp = r0, in0		// tmp = -dest
	;;
	and	loopcnt = 7, tmp		// loopcnt = -dest % 8
	cmp.ge	p6, p0 = OP_T_THRES, len	// is len <= OP_T_THRES
(p6)	br.cond.spnt	.cmpfew			// compare byte by byte
	;;
	cmp.eq	p6, p0 = loopcnt, r0
(p6)	br.cond.sptk .dest_aligned
	sub	len = len, loopcnt	// len -= -dest % 8
	adds	loopcnt = -1, loopcnt	// --loopcnt
	;;
	mov	ar.lc = loopcnt
.l1:					// copy -dest % 8 bytes
	ld1	value1 = [src], 1	// value = *src++
	ld1	value2 = [dest], 1
	;;
	cmp.ne	p6, p0 = value1, value2
(p6)	br.cond.spnt .done
	br.cloop.dptk .l1
.dest_aligned:
	and	sh1 = 7, src 		// sh1 = src % 8
	and	tmp = -8, len   	// tmp = len & -OPSIZ
	and	asrc = -8, src		// asrc = src & -OPSIZ  -- align src
	shr.u	loopcnt = len, 3	// loopcnt = len / 8
	and	len = 7, len ;;		// len = len % 8
	shl	sh1 = sh1, 3		// sh1 = 8 * (src % 8)
	adds	loopcnt = -1, loopcnt	// --loopcnt
	mov     pr.rot = 1 << 16 ;;	// set rotating predicates
	sub	sh2 = 64, sh1		// sh2 = 64 - sh1
	mov	ar.lc = loopcnt		// set LC
	cmp.eq  p6, p0 = sh1, r0 	// is the src aligned?
(p6)    br.cond.sptk .src_aligned
	add	src = src, tmp		// src += len & -OPSIZ
	mov	ar.ec = MEMLAT + 4 + 1 	// four more passes needed
	ld8	r[1] = [asrc], 8 ;;	// r[1] = w0
	.align	32

// We enter this loop with p6 cleared by the above comparison

.l2:
(p[0])		ld8	r[0] = [asrc], 8		// r[0] = w1
(p[0])		ld8	q[0] = [dest], 8
(p[MEMLAT])	shr.u	tmp1[0] = r[1 + MEMLAT], sh1	// tmp1 = w0 >> sh1
(p[MEMLAT])	shl	tmp2[0] = r[0 + MEMLAT], sh2  	// tmp2 = w1 << sh2
(p[MEMLAT+4])	cmp.ne	p6, p0 = q[MEMLAT + 4], val[1]
(p[MEMLAT+3])	or	val[0] = tmp1[3], tmp2[3] 	// val = tmp1 | tmp2
(p6)		br.cond.spnt .l2exit
		br.ctop.sptk    .l2
		br.cond.sptk .cmpfew
.l3exit:
	mux1	value1 = r[MEMLAT], @rev
	mux1	value2 = q[MEMLAT], @rev
	cmp.ne	p6, p0 = r0, r0	;;	// clear p6
.l2exit:
(p6)	mux1	value1 = val[1], @rev
(p6)	mux1	value2 = q[MEMLAT + 4], @rev ;;
	cmp.ltu	p6, p7 = value2, value1 ;;
(p6)	mov	ret0 = -1
(p7)	mov	ret0 = 1
	mov     pr = saved_pr, -1    	// restore the predicate registers
	mov 	ar.lc = saved_lc	// restore the loop counter
	br.ret.sptk.many b0
.src_aligned:
	cmp.ne	p6, p0 = r0, r0		// clear p6
	mov     ar.ec = MEMLAT + 1 ;;	// set EC
.l3:
(p[0])		ld8	r[0] = [src], 8
(p[0])		ld8	q[0] = [dest], 8
(p[MEMLAT])	cmp.ne	p6, p0 = r[MEMLAT], q[MEMLAT]
(p6)		br.cond.spnt .l3exit
		br.ctop.dptk .l3 ;;
.cmpfew:
	cmp.eq	p6, p0 = len, r0	// is len == 0 ?
	adds	len = -1, len		// --len;
(p6)	br.cond.spnt	.restore_and_exit ;;
	mov	ar.lc = len
.l4:
	ld1	value1 = [src], 1
	ld1	value2 = [dest], 1
	;;
	cmp.ne	p6, p0 = value1, value2
(p6)	br.cond.spnt	.done
	br.cloop.dptk	.l4 ;;
.done:
(p6)	sub	ret0 = value2, value1	// don't execute it if falling thru
.restore_and_exit:
	mov     pr = saved_pr, -1    	// restore the predicate registers
	mov 	ar.lc = saved_lc	// restore the loop counter
	br.ret.sptk.many b0
END(memcmp)

weak_alias (memcmp, bcmp)
libc_hidden_builtin_def (memcmp)
