/* Optimized version of the memccpy() function.
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

/* Return: a pointer to the next byte after char in dest or NULL

   Inputs:
        in0:    dest
        in1:    src
        in2:    char
        in3:    byte count

   This implementation assumes little endian mode (UM.be = 0).

   This implementation assumes that it is safe to do read ahead
   in the src block, without getting beyond its limit.  */

#include <sysdep.h>
#undef ret

#define OP_T_THRES 	16
#define OPSIZ 		8

#define saved_pr	r17
#define saved_lc	r18
#define dest		r19
#define src		r20
#define len		r21
#define asrc		r22
#define tmp		r23
#define char		r24
#define charx8		r25
#define saved_ec	r26
#define sh2		r28
#define	sh1		r29
#define loopcnt		r30
#define	value		r31

#ifdef GAS_ALIGN_BREAKS_UNWIND_INFO
/* Manually force proper loop-alignment.  Note: be sure to
   double-check the code-layout after making any changes to
   this routine! */
# define ALIGN(n)	{ nop 0 }
#else
# define ALIGN(n)	.align n
#endif

ENTRY(memccpy)
	.prologue
	alloc 	r2 = ar.pfs, 4, 40 - 4, 0, 40

#include "softpipe.h"
	.rotr	r[MEMLAT + 7], tmp1[4], tmp2[4], val[4], tmp3[2], pos0[2]
	.rotp	p[MEMLAT + 6 + 1]

	mov	ret0 = r0		// return NULL if no match
	.save pr, saved_pr
	mov	saved_pr = pr		// save the predicate registers
	mov 	dest = in0		// dest
	.save ar.lc, saved_lc
        mov 	saved_lc = ar.lc	// save the loop counter
        mov 	saved_ec = ar.ec	// save the loop counter
	.body
	mov 	src = in1		// src
	extr.u	char = in2, 0, 8	// char
	mov	len = in3		// len
	sub	tmp = r0, in0		// tmp = -dest
	cmp.ne	p7, p0 = r0, r0		// clear p7
	;;
	and	loopcnt = 7, tmp	// loopcnt = -dest % 8
	cmp.ge	p6, p0 = OP_T_THRES, len	// is len <= OP_T_THRES
	mov	ar.ec = 0		// ec not guaranteed zero on entry
(p6)	br.cond.spnt	.cpyfew		// copy byte by byte
	;;
	cmp.eq	p6, p0 = loopcnt, r0
	mux1	charx8 = char, @brcst
(p6)	br.cond.sptk .dest_aligned
	sub	len = len, loopcnt	// len -= -dest % 8
	adds	loopcnt = -1, loopcnt	// --loopcnt
	;;
	mov	ar.lc = loopcnt
.l1:					// copy -dest % 8 bytes
	ld1	value = [src], 1	// value = *src++
	;;
	st1	[dest] = value, 1	// *dest++ = value
	cmp.eq	p6, p0 = value, char
(p6)	br.cond.spnt .foundit
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
(p6)    br.cond.sptk .src_aligned ;;
	add	src = src, tmp		// src += len & -OPSIZ
	mov	ar.ec = MEMLAT + 6 + 1 	// six more passes needed
	ld8	r[1] = [asrc], 8 	// r[1] = w0
	cmp.ne	p6, p0 = r0, r0	;;	// clear p6
	ALIGN(32)
.l2:
(p[0])		ld8.s	r[0] = [asrc], 8		// r[0] = w1
(p[MEMLAT])	shr.u	tmp1[0] = r[1 + MEMLAT], sh1	// tmp1 = w0 >> sh1
(p[MEMLAT])	shl	tmp2[0] = r[0 + MEMLAT], sh2  	// tmp2 = w1 << sh2
(p[MEMLAT+4])	xor	tmp3[0] = val[1], charx8
(p[MEMLAT+5])	czx1.r	pos0[0] = tmp3[1]
(p[MEMLAT+6])	chk.s	r[6 + MEMLAT], .recovery1	// our data isn't
							// valid - rollback!
(p[MEMLAT+6])	cmp.ne	p6, p0 = 8, pos0[1]
(p6)		br.cond.spnt	.gotit
(p[MEMLAT+6])	st8	[dest] = val[3], 8		// store val to dest
(p[MEMLAT+3])	or	val[0] = tmp1[3], tmp2[3] 	// val = tmp1 | tmp2
		br.ctop.sptk    .l2
		br.cond.sptk .cpyfew

.src_aligned:
		cmp.ne  p6, p0 = r0, r0			// clear p6
		mov     ar.ec = MEMLAT + 2 + 1 ;;	// set EC
.l3:
(p[0])		ld8.s	r[0] = [src], 8
(p[MEMLAT])	xor	tmp3[0] = r[MEMLAT], charx8
(p[MEMLAT+1])	czx1.r	pos0[0] = tmp3[1]
(p[MEMLAT+2])	cmp.ne	p7, p0 = 8, pos0[1]
(p[MEMLAT+2])	chk.s	r[MEMLAT+2], .recovery2
(p7)		br.cond.spnt	.gotit
.back2:
(p[MEMLAT+2])	st8	[dest] = r[MEMLAT+2], 8
		br.ctop.dptk .l3
.cpyfew:
	cmp.eq	p6, p0 = len, r0	// is len == 0 ?
	adds	len = -1, len		// --len;
(p6)	br.cond.spnt	.restore_and_exit ;;
	mov	ar.lc = len
.l4:
	ld1	value = [src], 1
	;;
	st1	[dest] = value, 1
	cmp.eq	p6, p0 = value, char
(p6)	br.cond.spnt .foundit
	br.cloop.dptk	.l4 ;;
.foundit:
(p6)	mov	ret0 = dest
.restore_and_exit:
	mov     pr = saved_pr, -1    	// restore the predicate registers
	mov 	ar.lc = saved_lc	// restore the loop counter
	mov 	ar.ec = saved_ec ;;	// restore the epilog counter
	br.ret.sptk.many b0
.gotit:
	.pred.rel "mutex" p6, p7
(p6)	mov	value = val[3]		// if coming from l2
(p7)	mov	value = r[MEMLAT+2]	// if coming from l3
	mov	ar.lc = pos0[1] ;;
.l5:
	extr.u	tmp = value, 0, 8 ;;
	st1	[dest] = tmp, 1
	shr.u	value = value, 8
	br.cloop.sptk .l5 ;;
	mov 	ret0 = dest
	mov	pr = saved_pr, -1
	mov	ar.lc = saved_lc
	br.ret.sptk.many b0

.recovery1:
#if MEMLAT != 6
# error "MEMLAT must be 6!"
#endif
	adds	src = -8, asrc
	mov	loopcnt = ar.lc
	mov	tmp = ar.ec
	;;
(p[0])	adds	src = -8, src
	;;
(p[1])	adds	src = -8, src
	sub	sh1 = (MEMLAT + 6 + 1), tmp
	;;
(p[2])	adds	src = -8, src
	;;
(p[3])	adds	src = -8, src
	shl	loopcnt = loopcnt, 3
	;;
(p[4])	adds	src = -8, src
	;;
(p[5])	adds	src = -8, src
	shl	sh1 = sh1, 3
	;;
(p[6])	adds	src = -8, src
	;;
(p[7])	adds	src = -8, src
	shl	tmp = tmp, 3
	;;
(p[8])	adds	src = -8, src
	;;
(p[9])	adds	src = -8, src
	shr.u	sh2 = sh2, 3
	;;
(p[10])	adds	src = -8, src
	;;
(p[11])	adds	src = -8, src
	add	len = len, loopcnt
	;;
	sub	src = src, sh2
	;;
	add	len = tmp, len
	add	src = sh1, src
	br.cond.sptk .cpyfew

.recovery2:
#if MEMLAT != 6
# error "MEMLAT must be 6!"
#endif
	add	tmp = -8, src
(p7)	br.cond.spnt .gotit
	;;
(p[0])	add	tmp = -8, tmp ;;
(p[1])	add	tmp = -8, tmp ;;
(p[2])	add	tmp = -8, tmp ;;
(p[3])	add	tmp = -8, tmp ;;
(p[4])	add	tmp = -8, tmp ;;
(p[5])	add	tmp = -8, tmp ;;
(p[6])	add	tmp = -8, tmp ;;
(p[7])	add	tmp = -8, tmp ;;
	ld8	r[MEMLAT+2] = [tmp] ;;
	xor	pos0[1] = r[MEMLAT+2], charx8 ;;
	czx1.r	pos0[1] = pos0[1] ;;
	cmp.ne	p7, p6 = 8, pos0[1]
(p7)	br.cond.spnt .gotit
	br.cond.sptk .back2
END(memccpy)
