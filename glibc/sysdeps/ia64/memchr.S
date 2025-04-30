/* Optimized version of the standard memchr() function.
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
	in0:	str
	in1:	chr
	in2:	byte count

   This implementation assumes little endian mode.  For big endian mode,
   the instruction czx1.r should be replaced by czx1.l.

   The algorithm is fairly straightforward: search byte by byte until we
   we get to a word aligned address, then search word by word as much as
   possible; the remaining few bytes are searched one at a time.

   The word by word search is performed by xor-ing the word with a word
   containing chr in every byte.  If there is a hit, the result will
   contain a zero byte in the corresponding position.  The presence and
   position of that zero byte is detected with a czx instruction.

   All the loops in this function could have had the internal branch removed
   if br.ctop and br.cloop could be predicated :-(.  */

#include <sysdep.h>
#undef ret

#define saved_pr	r15
#define saved_lc	r16
#define	chr		r17
#define len		r18
#define last		r20
#define val		r21
#define tmp		r24
#define chrx8		r25
#define loopcnt		r30

#define str		in0

ENTRY(__memchr)
	.prologue
	alloc r2 = ar.pfs, 3, 0, 29, 32
#include "softpipe.h"
	.rotr	value[MEMLAT+1], addr[MEMLAT+3], aux[2], poschr[2]
	.rotp	p[MEMLAT+3]
	.save ar.lc, saved_lc
	mov	saved_lc = ar.lc	// save the loop counter
	.save pr, saved_pr
	mov	saved_pr = pr		// save the predicates
	.body
	mov	ret0 = str
	add	last = str, in2		// last byte
	;;
	cmp.ltu	p6, p0 = last, str
	;;
(p6)	mov	last = -1
	and	tmp = 7, str		// tmp = str % 8
	cmp.ne	p7, p0 = r0, r0		// clear p7
	extr.u	chr = in1, 0, 8		// chr = (unsigned char) in1
	mov	len = in2
	cmp.gtu	p6, p0 = 16, in2	// use a simple loop for short
(p6)	br.cond.spnt .srchfew ;;	// searches
	sub	loopcnt = 8, tmp	// loopcnt = 8 - tmp
	cmp.eq	p6, p0 = tmp, r0
(p6)	br.cond.sptk	.str_aligned;;
	sub	len = len, loopcnt
	adds	loopcnt = -1, loopcnt;;
	mov	ar.lc = loopcnt
.l1:
	ld1	val = [ret0], 1
	;;
	cmp.eq	p6, p0 = val, chr
(p6)	br.cond.spnt	.foundit
	br.cloop.sptk	.l1 ;;
.str_aligned:
	cmp.ne	p6, p0 = r0, r0		// clear p6
	shr.u	loopcnt = len, 3	// loopcnt = len / 8
	and	len = 7, len ;;		// remaining len = len & 7
	adds	loopcnt = -1, loopcnt
	mov	ar.ec = MEMLAT + 3
	mux1	chrx8 = chr, @brcst ;;	// get a word full of chr
	mov	ar.lc = loopcnt
	mov	pr.rot = 1 << 16 ;;
.l2:
(p[0])		mov	addr[0] = ret0
(p[0])		ld8.s	value[0] = [ret0], 8	 // speculative load
(p[MEMLAT])	chk.s	value[MEMLAT], .recovery // check and recovery
(p[MEMLAT])	xor	aux[0] = value[MEMLAT], chrx8
(p[MEMLAT+1])	czx1.r	poschr[0] = aux[1]
(p[MEMLAT+2])	cmp.ne	p7, p0 = 8, poschr[1]
(p7)		br.cond.dpnt .foundit
		br.ctop.dptk .l2
.srchfew:
	adds	loopcnt = -1, len
	cmp.eq	p6, p0 = len, r0
(p6)	br.cond.spnt .notfound ;;
	mov	ar.lc = loopcnt
.l3:
	ld1	val = [ret0], 1
	;;
	cmp.eq	p6, p0 = val, chr
(p6)	br.cond.dpnt	.foundit
	br.cloop.sptk	.l3 ;;
.notfound:
	cmp.ne	p6, p0 = r0, r0	// clear p6 (p7 was already 0 when we got here)
	mov	ret0 = r0 ;;	// return NULL
.foundit:
	.pred.rel "mutex" p6, p7
(p6)	adds	ret0 = -1, ret0			   // if we got here from l1 or l3
(p7)	add	ret0 = addr[MEMLAT+2], poschr[1]   // if we got here from l2
	mov	pr = saved_pr, -1
	mov	ar.lc = saved_lc
	br.ret.sptk.many b0

.recovery:
#if MEMLAT != 6
# error "MEMLAT must be 6!"
#endif
(p[MEMLAT-6])	add	ret0 = -8, ret0;;
(p[MEMLAT-5])	add	ret0 = -8, ret0;;
(p[MEMLAT-4])	add	ret0 = -8, ret0;;
(p[MEMLAT-3])	add	ret0 = -8, ret0;;
(p[MEMLAT-2])	add	ret0 = -8, ret0;;
(p[MEMLAT-1])	add	ret0 = -8, ret0;;
(p[MEMLAT])	add	ret0 = -8, ret0;;
(p[MEMLAT+1])	add	ret0 = -8, ret0;;
(p[MEMLAT+2])	add	ret0 = -8, ret0;;
.l4:
	mov     addr[MEMLAT+2] = ret0
	ld8	tmp = [ret0];;		// load the first unchecked 8byte
	xor	aux[1] = tmp, chrx8;;
	czx1.r	poschr[1] = aux[1];;
	cmp.ne	p7, p0 = 8, poschr[1];;
(p7)	add	ret0 = addr[MEMLAT+2], poschr[1];;
(p7)	cmp.geu	p6, p7 = ret0, last	// don't go over the last byte
(p6)	br.cond.spnt	.notfound;;
(p7)	br.cond.spnt	.foundit;;
	adds	ret0 = 8, ret0		// load the next unchecked 8byte
	br.sptk	.l4;;

END(__memchr)

weak_alias (__memchr, memchr)
libc_hidden_builtin_def (memchr)
