/* Optimized version of the standard bzero() function.
   This file is part of the GNU C Library.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   Contributed by Dan Pop for Itanium <Dan.Pop@cern.ch>.
   Rewritten for McKinley by Sverre Jarp, HP Labs/CERN <Sverre.Jarp@cern.ch>

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
        in1:    count

   The algorithm is fairly straightforward: set byte by byte until we
   we get to a 16B-aligned address, then loop on 128 B chunks using an
   early store as prefetching, then loop on 32B chucks, then clear remaining
   words, finally clear remaining bytes.
   Since a stf.spill f0 can store 16B in one go, we use this instruction
   to get peak speed.  */

#include <sysdep.h>
#undef ret

#define dest		in0
#define	cnt		in1

#define tmp		r31
#define save_lc		r30
#define ptr0		r29
#define ptr1		r28
#define ptr2		r27
#define ptr3		r26
#define ptr9 		r24
#define	loopcnt		r23
#define linecnt		r22
#define bytecnt		r21

// This routine uses only scratch predicate registers (p6 - p15)
#define p_scr		p6	// default register for same-cycle branches
#define p_unalgn	p9
#define p_y		p11
#define p_n		p12
#define p_yy		p13
#define p_nn		p14

#define movi0		mov

#define MIN1		15
#define MIN1P1HALF	8
#define LINE_SIZE	128
#define LSIZE_SH        7			// shift amount
#define PREF_AHEAD	8

#define USE_FLP
#if defined(USE_INT)
#define store		st8
#define myval		r0
#elif defined(USE_FLP)
#define store		stf8
#define myval		f0
#endif

.align	64
ENTRY(bzero)
{ .mmi
	.prologue
	alloc	tmp = ar.pfs, 2, 0, 0, 0
	lfetch.nt1 [dest]
	.save   ar.lc, save_lc
	movi0	save_lc = ar.lc
} { .mmi
	.body
	mov	ret0 = dest		// return value
	nop.m	0
	cmp.eq	p_scr, p0 = cnt, r0
;; }
{ .mmi
	and	ptr2 = -(MIN1+1), dest	// aligned address
	and	tmp = MIN1, dest	// prepare to check for alignment
	tbit.nz p_y, p_n = dest, 0	// Do we have an odd address? (M_B_U)
} { .mib
	mov	ptr1 = dest
	nop.i	0
(p_scr)	br.ret.dpnt.many rp		// return immediately if count = 0
;; }
{ .mib
	cmp.ne	p_unalgn, p0 = tmp, r0
} { .mib					// NB: # of bytes to move is 1
	sub	bytecnt = (MIN1+1), tmp		//     higher than loopcnt
	cmp.gt	p_scr, p0 = 16, cnt		// is it a minimalistic task?
(p_scr)	br.cond.dptk.many .move_bytes_unaligned	// go move just a few (M_B_U)
;; }
{ .mmi
(p_unalgn) add	ptr1 = (MIN1+1), ptr2		// after alignment
(p_unalgn) add	ptr2 = MIN1P1HALF, ptr2		// after alignment
(p_unalgn) tbit.nz.unc p_y, p_n = bytecnt, 3	// should we do a st8 ?
;; }
{ .mib
(p_y)	add	cnt = -8, cnt
(p_unalgn) tbit.nz.unc p_yy, p_nn = bytecnt, 2	// should we do a st4 ?
} { .mib
(p_y)	st8	[ptr2] = r0,-4
(p_n)	add	ptr2 = 4, ptr2
;; }
{ .mib
(p_yy)	add	cnt = -4, cnt
(p_unalgn) tbit.nz.unc p_y, p_n = bytecnt, 1	// should we do a st2 ?
} { .mib
(p_yy)	st4	[ptr2] = r0,-2
(p_nn)	add	ptr2 = 2, ptr2
;; }
{ .mmi
	mov	tmp = LINE_SIZE+1		// for compare
(p_y)	add	cnt = -2, cnt
(p_unalgn) tbit.nz.unc p_yy, p_nn = bytecnt, 0	// should we do a st1 ?
} { .mmi
	nop.m	0
(p_y)	st2	[ptr2] = r0,-1
(p_n)	add	ptr2 = 1, ptr2
;; }

{ .mmi
(p_yy)	st1	[ptr2] = r0
	cmp.gt	p_scr, p0 = tmp, cnt		// is it a minimalistic task?
} { .mbb
(p_yy)	add	cnt = -1, cnt
(p_scr)	br.cond.dpnt.many .fraction_of_line	// go move just a few
;; }
{ .mib
	nop.m 	0
	shr.u	linecnt = cnt, LSIZE_SH
	nop.b	0
;; }

	.align 32
.l1b:	// ------------------//  L1B: store ahead into cache lines; fill later
{ .mmi
	and	tmp = -(LINE_SIZE), cnt		// compute end of range
	mov	ptr9 = ptr1			// used for prefetching
	and	cnt = (LINE_SIZE-1), cnt	// remainder
} { .mmi
	mov	loopcnt = PREF_AHEAD-1		// default prefetch loop
	cmp.gt	p_scr, p0 = PREF_AHEAD, linecnt	// check against actual value
;; }
{ .mmi
(p_scr)	add	loopcnt = -1, linecnt
	add	ptr2 = 16, ptr1	// start of stores (beyond prefetch stores)
	add	ptr1 = tmp, ptr1	// first address beyond total range
;; }
{ .mmi
	add	tmp = -1, linecnt	// next loop count
	movi0	ar.lc = loopcnt
;; }
.pref_l1b:
{ .mib
	stf.spill [ptr9] = f0, 128	// Do stores one cache line apart
	nop.i   0
	br.cloop.dptk.few .pref_l1b
;; }
{ .mmi
	add	ptr0 = 16, ptr2		// Two stores in parallel
	movi0	ar.lc = tmp
;; }
.l1bx:
 { .mmi
	stf.spill [ptr2] = f0, 32
	stf.spill [ptr0] = f0, 32
 ;; }
 { .mmi
	stf.spill [ptr2] = f0, 32
	stf.spill [ptr0] = f0, 32
 ;; }
 { .mmi
	stf.spill [ptr2] = f0, 32
	stf.spill [ptr0] = f0, 64
	cmp.lt	p_scr, p0 = ptr9, ptr1	// do we need more prefetching?
 ;; }
{ .mmb
	stf.spill [ptr2] = f0, 32
(p_scr)	stf.spill [ptr9] = f0, 128
	br.cloop.dptk.few .l1bx
;; }
{ .mib
	cmp.gt  p_scr, p0 = 8, cnt	// just a few bytes left ?
(p_scr)	br.cond.dpnt.many  .move_bytes_from_alignment
;; }

.fraction_of_line:
{ .mib
	add	ptr2 = 16, ptr1
	shr.u	loopcnt = cnt, 5   	// loopcnt = cnt / 32
;; }
{ .mib
	cmp.eq	p_scr, p0 = loopcnt, r0
	add	loopcnt = -1, loopcnt
(p_scr)	br.cond.dpnt.many .store_words
;; }
{ .mib
	and	cnt = 0x1f, cnt		// compute the remaining cnt
	movi0   ar.lc = loopcnt
;; }
	.align 32
.l2:	// -----------------------------//  L2A:  store 32B in 2 cycles
{ .mmb
	store	[ptr1] = myval, 8
	store	[ptr2] = myval, 8
;; } { .mmb
	store	[ptr1] = myval, 24
	store	[ptr2] = myval, 24
	br.cloop.dptk.many .l2
;; }
.store_words:
{ .mib
	cmp.gt	p_scr, p0 = 8, cnt	// just a few bytes left ?
(p_scr)	br.cond.dpnt.many .move_bytes_from_alignment	// Branch
;; }

{ .mmi
	store	[ptr1] = myval, 8	// store
	cmp.le	p_y, p_n = 16, cnt	//
	add	cnt = -8, cnt		// subtract
;; }
{ .mmi
(p_y)	store	[ptr1] = myval, 8	// store
(p_y)	cmp.le.unc p_yy, p_nn = 16, cnt
(p_y)	add	cnt = -8, cnt		// subtract
;; }
{ .mmi					// store
(p_yy)	store	[ptr1] = myval, 8
(p_yy)	add	cnt = -8, cnt		// subtract
;; }

.move_bytes_from_alignment:
{ .mib
	cmp.eq	p_scr, p0 = cnt, r0
	tbit.nz.unc p_y, p0 = cnt, 2	// should we terminate with a st4 ?
(p_scr)	br.cond.dpnt.few .restore_and_exit
;; }
{ .mib
(p_y)	st4	[ptr1] = r0,4
	tbit.nz.unc p_yy, p0 = cnt, 1	// should we terminate with a st2 ?
;; }
{ .mib
(p_yy)	st2	[ptr1] = r0,2
	tbit.nz.unc p_y, p0 = cnt, 0	// should we terminate with a st1 ?
;; }

{ .mib
(p_y)	st1	[ptr1] = r0
;; }
.restore_and_exit:
{ .mib
	nop.m	0
	movi0	ar.lc = save_lc
	br.ret.sptk.many rp
;; }

.move_bytes_unaligned:
{ .mmi
       .pred.rel "mutex",p_y, p_n
       .pred.rel "mutex",p_yy, p_nn
(p_n)	cmp.le  p_yy, p_nn = 4, cnt
(p_y)	cmp.le  p_yy, p_nn = 5, cnt
(p_n)	add	ptr2 = 2, ptr1
} { .mmi
(p_y)	add	ptr2 = 3, ptr1
(p_y)	st1	[ptr1] = r0, 1		// fill 1 (odd-aligned) byte
(p_y)	add	cnt = -1, cnt		// [15, 14 (or less) left]
;; }
{ .mmi
(p_yy)	cmp.le.unc p_y, p0 = 8, cnt
	add	ptr3 = ptr1, cnt	// prepare last store
	movi0	ar.lc = save_lc
} { .mmi
(p_yy)	st2	[ptr1] = r0, 4		// fill 2 (aligned) bytes
(p_yy)	st2	[ptr2] = r0, 4		// fill 2 (aligned) bytes
(p_yy)	add	cnt = -4, cnt		// [11, 10 (o less) left]
;; }
{ .mmi
(p_y)	cmp.le.unc p_yy, p0 = 8, cnt
	add	ptr3 = -1, ptr3		// last store
	tbit.nz p_scr, p0 = cnt, 1	// will there be a st2 at the end ?
} { .mmi
(p_y)	st2	[ptr1] = r0, 4		// fill 2 (aligned) bytes
(p_y)	st2	[ptr2] = r0, 4		// fill 2 (aligned) bytes
(p_y)	add	cnt = -4, cnt		// [7, 6 (or less) left]
;; }
{ .mmi
(p_yy)	st2	[ptr1] = r0, 4		// fill 2 (aligned) bytes
(p_yy)	st2	[ptr2] = r0, 4		// fill 2 (aligned) bytes
					// [3, 2 (or less) left]
	tbit.nz p_y, p0 = cnt, 0	// will there be a st1 at the end ?
} { .mmi
(p_yy)	add	cnt = -4, cnt
;; }
{ .mmb
(p_scr)	st2	[ptr1] = r0		// fill 2 (aligned) bytes
(p_y)	st1	[ptr3] = r0		// fill last byte (using ptr3)
	br.ret.sptk.many rp
;; }
END(bzero)
