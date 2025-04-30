/* Optimized version of the standard memcpy() function.
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
        in1:    src
        in2:    byte count

   An assembly implementation of the algorithm used by the generic C
   version from glibc.  The case when source and sest are aligned is
   treated separately, for extra performance.

   In this form, memcpy assumes little endian mode.  For big endian mode,
   sh1 must be computed using an extra instruction: sub sh1 = 64, sh1
   and the order of r[MEMLAT] and r[MEMLAT+1] must be reverted in the
   shrp instruction.  */

#define USE_LFETCH
#define USE_FLP
#include <sysdep.h>
#undef ret

#define LFETCH_DIST     500

#define ALIGN_UNROLL_no   4 // no. of elements
#define ALIGN_UNROLL_sh	  2 // (shift amount)

#define MEMLAT	8
#define Nrot	((4*(MEMLAT+2) + 7) & ~7)

#define OP_T_THRES 	16
#define OPSIZ 		8

#define loopcnt		r14
#define elemcnt		r15
#define saved_pr	r16
#define saved_lc	r17
#define adest		r18
#define dest		r19
#define asrc		r20
#define src		r21
#define len		r22
#define tmp2		r23
#define tmp3		r24
#define	tmp4		r25
#define ptable		r26
#define ploop56		r27
#define	loopaddr	r28
#define	sh1		r29
#define ptr1		r30
#define ptr2		r31

#define movi0 		mov

#define p_scr		p6
#define p_xtr		p7
#define p_nxtr		p8
#define p_few		p9

#if defined(USE_FLP)
#define load		ldf8
#define store		stf8
#define tempreg		f6
#define the_r		fr
#define the_s		fs
#define the_t		ft
#define the_q		fq
#define the_w		fw
#define the_x		fx
#define the_y		fy
#define the_z		fz
#elif defined(USE_INT)
#define load		ld8
#define store		st8
#define tempreg		tmp2
#define the_r		r
#define the_s		s
#define the_t		t
#define the_q		q
#define the_w		w
#define the_x		x
#define the_y		y
#define the_z		z
#endif

#ifdef GAS_ALIGN_BREAKS_UNWIND_INFO
/* Manually force proper loop-alignment.  Note: be sure to
   double-check the code-layout after making any changes to
   this routine! */
# define ALIGN(n)	{ nop 0 }
#else
# define ALIGN(n)	.align n
#endif

#if defined(USE_LFETCH)
#define LOOP(shift)						\
		ALIGN(32);					\
.loop##shift##:							\
{ .mmb								\
(p[0])	ld8.nt1	r[0] = [asrc], 8 ;				\
(p[0])	lfetch.nt1 [ptr1], 16 ;					\
	nop.b 0 ;						\
} { .mib							\
(p[MEMLAT+1]) st8 [dest] = tmp3, 8 ;				\
(p[MEMLAT]) shrp tmp3 = r[MEMLAT], s[MEMLAT+1], shift ;		\
	nop.b 0 ;;						\
 } { .mmb							\
(p[0])	ld8.nt1	s[0] = [asrc], 8 ;				\
(p[0])	lfetch.nt1	[ptr2], 16 ;				\
	nop.b 0 ;						\
} { .mib							\
(p[MEMLAT+1]) st8 [dest] = tmp4, 8 ;				\
(p[MEMLAT]) shrp tmp4 = s[MEMLAT], r[MEMLAT], shift ;		\
	br.ctop.sptk.many .loop##shift 				\
;; }								\
{ .mib								\
	br.cond.sptk.many .copy_bytes ; /* deal with the remaining bytes */  \
}
#else
#define LOOP(shift)						\
		ALIGN(32);					\
.loop##shift##:							\
{ .mmb								\
(p[0])	ld8.nt1	r[0] = [asrc], 8 ;				\
	nop.b 0 ;						\
} { .mib							\
(p[MEMLAT+1]) st8 [dest] = tmp3, 8 ;				\
(p[MEMLAT]) shrp tmp3 = r[MEMLAT], s[MEMLAT+1], shift ;		\
	nop.b 0 ;;						\
 } { .mmb							\
(p[0])	ld8.nt1	s[0] = [asrc], 8 ;				\
	nop.b 0 ;						\
} { .mib							\
(p[MEMLAT+1]) st8 [dest] = tmp4, 8 ;				\
(p[MEMLAT]) shrp tmp4 = s[MEMLAT], r[MEMLAT], shift ;		\
	br.ctop.sptk.many .loop##shift 				\
;; }								\
{ .mib								\
	br.cond.sptk.many .copy_bytes ; /* deal with the remaining bytes */  \
}
#endif


ENTRY(memcpy)
{ .mmi
	.prologue
	alloc 	r2 = ar.pfs, 3, Nrot - 3, 0, Nrot
	.rotr	r[MEMLAT+1], s[MEMLAT+2], q[MEMLAT+1], t[MEMLAT+1]
	.rotp	p[MEMLAT+2]
	.rotf	fr[MEMLAT+1], fq[MEMLAT+1], fs[MEMLAT+1], ft[MEMLAT+1]
	mov	ret0 = in0		// return tmp2 = dest
	.save   pr, saved_pr
	movi0	saved_pr = pr		// save the predicate registers
} { .mmi
	and	tmp4 = 7, in0 		// check if destination is aligned
	mov 	dest = in0		// dest
	mov 	src = in1		// src
;; }
{ .mii
	cmp.eq	p_scr, p0 = in2, r0	// if (len == 0)
	.save   ar.lc, saved_lc
        movi0 	saved_lc = ar.lc	// save the loop counter
	.body
	cmp.ge	p_few, p0 = OP_T_THRES, in2 // is len <= OP_T_THRESH
} { .mbb
	mov	len = in2		// len
(p_scr)	br.cond.dpnt.few .restore_and_exit // 	Branch no. 1: return dest
(p_few) br.cond.dpnt.many .copy_bytes	// Branch no. 2: copy byte by byte
;; }
{ .mmi
#if defined(USE_LFETCH)
	lfetch.nt1 [dest]		//
	lfetch.nt1 [src]		//
#endif
	shr.u	elemcnt = len, 3	// elemcnt = len / 8
} { .mib
	cmp.eq	p_scr, p0 = tmp4, r0	// is destination aligned?
	sub	loopcnt = 7, tmp4	//
(p_scr) br.cond.dptk.many .dest_aligned
;; }
{ .mmi
	ld1	tmp2 = [src], 1		//
	sub	len = len, loopcnt, 1	// reduce len
	movi0	ar.lc = loopcnt		//
} { .mib
	cmp.ne  p_scr, p0 = 0, loopcnt	// avoid loading beyond end-point
;; }

.l0:	// ---------------------------- // L0: Align src on 8-byte boundary
{ .mmi
	st1	[dest] = tmp2, 1	//
(p_scr)	ld1	tmp2 = [src], 1		//
} { .mib
	cmp.lt	p_scr, p0 = 1, loopcnt	// avoid load beyond end-point
	add	loopcnt = -1, loopcnt
	br.cloop.dptk.few .l0		//
;; }

.dest_aligned:
{ .mmi
	and	tmp4 = 7, src		// ready for alignment check
	shr.u	elemcnt = len, 3	// elemcnt = len / 8
;; }
{ .mib
	cmp.ne	p_scr, p0 = tmp4, r0	// is source also aligned
	tbit.nz p_xtr, p_nxtr = src, 3	// prepare a separate move if src
} { .mib				// is not 16B aligned
	add	ptr2 = LFETCH_DIST, dest	// prefetch address
	add	ptr1 = LFETCH_DIST, src
(p_scr) br.cond.dptk.many .src_not_aligned
;; }

// The optimal case, when dest, and src are aligned

.both_aligned:
{ .mmi
	.pred.rel "mutex",p_xtr,p_nxtr
(p_xtr)	cmp.gt  p_scr, p0 = ALIGN_UNROLL_no+1, elemcnt // Need N + 1 to qualify
(p_nxtr) cmp.gt p_scr, p0 = ALIGN_UNROLL_no, elemcnt  // Need only N to qualify
	movi0	pr.rot = 1 << 16	// set rotating predicates
} { .mib
(p_scr) br.cond.dpnt.many .copy_full_words
;; }

{ .mmi
(p_xtr)	load	tempreg = [src], 8
(p_xtr) add 	elemcnt = -1, elemcnt
	movi0	ar.ec = MEMLAT + 1	// set the epilog counter
;; }
{ .mmi
(p_xtr) add	len = -8, len		//
	add 	asrc = 16, src 		// one bank apart (for USE_INT)
	shr.u	loopcnt = elemcnt, ALIGN_UNROLL_sh  // cater for unrolling
;;}
{ .mmi
	add	loopcnt = -1, loopcnt
(p_xtr)	store	[dest] = tempreg, 8	// copy the "extra" word
	nop.i	0
;; }
{ .mib
	add	adest = 16, dest
	movi0	ar.lc = loopcnt 	// set the loop counter
;; }

#ifdef  GAS_ALIGN_BREAKS_UNWIND_INFO
	{ nop 0 }
#else
	.align	32
#endif
#if defined(USE_FLP)
.l1: // ------------------------------- // L1: Everything a multiple of 8
{ .mmi
#if defined(USE_LFETCH)
(p[0])	lfetch.nt1 [ptr2],32
#endif
(p[0])	ldfp8	the_r[0],the_q[0] = [src], 16
(p[0])	add	len = -32, len
} {.mmb
(p[MEMLAT]) store [dest] = the_r[MEMLAT], 8
(p[MEMLAT]) store [adest] = the_s[MEMLAT], 8
;; }
{ .mmi
#if defined(USE_LFETCH)
(p[0])	lfetch.nt1 [ptr1],32
#endif
(p[0])	ldfp8	the_s[0], the_t[0] = [src], 16
} {.mmb
(p[MEMLAT]) store [dest] = the_q[MEMLAT], 24
(p[MEMLAT]) store [adest] = the_t[MEMLAT], 24
	br.ctop.dptk.many .l1
;; }
#elif defined(USE_INT)
.l1: // ------------------------------- // L1: Everything a multiple of 8
{ .mmi
(p[0])	load	the_r[0] = [src], 8
(p[0])	load	the_q[0] = [asrc], 8
(p[0])	add	len = -32, len
} {.mmb
(p[MEMLAT]) store [dest] = the_r[MEMLAT], 8
(p[MEMLAT]) store [adest] = the_q[MEMLAT], 8
;; }
{ .mmi
(p[0])	load	the_s[0]  = [src], 24
(p[0])	load	the_t[0] = [asrc], 24
} {.mmb
(p[MEMLAT]) store [dest] = the_s[MEMLAT], 24
(p[MEMLAT]) store [adest] = the_t[MEMLAT], 24
#if defined(USE_LFETCH)
;; }
{ .mmb
(p[0])	lfetch.nt1 [ptr2],32
(p[0])	lfetch.nt1 [ptr1],32
#endif
	br.ctop.dptk.many .l1
;; }
#endif

.copy_full_words:
{ .mib
	cmp.gt	p_scr, p0 = 8, len	//
	shr.u	elemcnt = len, 3	//
(p_scr) br.cond.dpnt.many .copy_bytes
;; }
{ .mii
	load	tempreg = [src], 8
	add	loopcnt = -1, elemcnt	//
;; }
{ .mii
	cmp.ne	p_scr, p0 = 0, loopcnt	//
	mov	ar.lc = loopcnt		//
;; }

.l2: // ------------------------------- // L2: Max 4 words copied separately
{ .mmi
	store	[dest] = tempreg, 8
(p_scr)	load	tempreg = [src], 8	//
	add	len = -8, len
} { .mib
	cmp.lt	p_scr, p0 = 1, loopcnt	// avoid load beyond end-point
	add	loopcnt = -1, loopcnt
	br.cloop.dptk.few  .l2
;; }

.copy_bytes:
{ .mib
	cmp.eq	p_scr, p0 = len, r0	// is len == 0 ?
	add	loopcnt = -1, len	// len--;
(p_scr)	br.cond.spnt	.restore_and_exit
;; }
{ .mii
	ld1	tmp2 = [src], 1
	movi0	ar.lc = loopcnt
	cmp.ne	p_scr, p0 = 0, loopcnt	// avoid load beyond end-point
;; }

.l3: // ------------------------------- // L3: Final byte move
{ .mmi
	st1	[dest] = tmp2, 1
(p_scr)	ld1	tmp2 = [src], 1
} { .mib
	cmp.lt	p_scr, p0 = 1, loopcnt	// avoid load beyond end-point
	add	loopcnt = -1, loopcnt
	br.cloop.dptk.few  .l3
;; }

.restore_and_exit:
{ .mmi
	movi0	pr = saved_pr, -1	// restore the predicate registers
;; }
{ .mib
	movi0	ar.lc = saved_lc	// restore the loop counter
	br.ret.sptk.many b0
;; }


.src_not_aligned:
{ .mmi
	cmp.gt	p_scr, p0 = 16, len
	and	sh1 = 7, src 		// sh1 = src % 8
	shr.u	loopcnt = len, 4	// element-cnt = len / 16
} { .mib
	add	tmp4 = @ltoff(.table), gp
	add 	tmp3 = @ltoff(.loop56), gp
(p_scr)	br.cond.dpnt.many .copy_bytes	// do byte by byte if too few
;; }
{ .mmi
	and	asrc = -8, src		// asrc = (-8) -- align src for loop
	add 	loopcnt = -1, loopcnt	// loopcnt--
	shl	sh1 = sh1, 3		// sh1 = 8 * (src % 8)
} { .mmi
	ld8	ptable = [tmp4]		// ptable = &table
	ld8	ploop56 = [tmp3]	// ploop56 = &loop56
	and	tmp2 = -16, len		// tmp2 = len & -OPSIZ
;; }
{ .mmi
	add	tmp3 = ptable, sh1	// tmp3 = &table + sh1
	add	src = src, tmp2		// src += len & (-16)
	movi0	ar.lc = loopcnt		// set LC
;; }
{ .mmi
	ld8	tmp4 = [tmp3]		// tmp4 = loop offset
	sub	len = len, tmp2		// len -= len & (-16)
	movi0	ar.ec = MEMLAT + 2 	// one more pass needed
;; }
{ .mmi
	ld8	s[1] = [asrc], 8	// preload
	sub	loopaddr = ploop56,tmp4	// loopadd = &loop56 - loop offset
	movi0   pr.rot = 1 << 16	// set rotating predicates
;; }
{ .mib
	nop.m	0
	movi0	b6 = loopaddr
	br	b6			// jump to the appropriate loop
;; }

	LOOP(8)
	LOOP(16)
	LOOP(24)
	LOOP(32)
	LOOP(40)
	LOOP(48)
	LOOP(56)
END(memcpy)
libc_hidden_builtin_def (memcpy)

	.rodata
	.align 8
.table:
	data8	0			// dummy entry
	data8 	.loop56 - .loop8
	data8 	.loop56 - .loop16
	data8 	.loop56 - .loop24
	data8	.loop56 - .loop32
	data8	.loop56 - .loop40
	data8	.loop56 - .loop48
	data8	.loop56 - .loop56
