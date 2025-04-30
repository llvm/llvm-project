/* Optimized version of the standard memmove() function.
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
        in2:    byte count

   The core of the function is the memcpy implementation used in memcpy.S.
   When bytes have to be copied backwards, only the easy case, when
   all arguments are multiples of 8, is optimised.

   In this form, it assumes little endian mode.  For big endian mode,
   sh1 must be computed using an extra instruction: sub sh1 = 64, sh1
   or the UM.be bit should be cleared at the beginning and set at the end.  */

#include <sysdep.h>
#undef ret

#define OP_T_THRES 	16
#define OPSIZ 		 8

#define adest		r15
#define saved_pr	r17
#define saved_lc	r18
#define dest		r19
#define src		r20
#define len		r21
#define asrc		r22
#define tmp2		r23
#define tmp3		r24
#define	tmp4		r25
#define ptable		r26
#define ploop56		r27
#define	loopaddr	r28
#define	sh1		r29
#define loopcnt		r30
#define	value		r31

#ifdef GAS_ALIGN_BREAKS_UNWIND_INFO
# define ALIGN(n)	{ nop 0 }
#else
# define ALIGN(n)	.align n
#endif

#define LOOP(shift)							\
		ALIGN(32);						\
.loop##shift##:								\
(p[0])		ld8	r[0] = [asrc], 8 ;	/* w1 */		\
(p[MEMLAT+1])	st8	[dest] = value, 8 ;				\
(p[MEMLAT])	shrp	value = r[MEMLAT], r[MEMLAT+1], shift ;		\
		nop.b	0 ;						\
		nop.b	0 ;						\
		br.ctop.sptk .loop##shift ;				\
		br.cond.sptk .cpyfew ; /* deal with the remaining bytes */

#define MEMLAT	21
#define Nrot	(((2*MEMLAT+3) + 7) & ~7)

ENTRY(memmove)
	.prologue
	alloc 	r2 = ar.pfs, 3, Nrot - 3, 0, Nrot
	.rotr	r[MEMLAT + 2], q[MEMLAT + 1]
	.rotp	p[MEMLAT + 2]
	mov	ret0 = in0		// return value = dest
	.save pr, saved_pr
	mov	saved_pr = pr		// save the predicate registers
	.save ar.lc, saved_lc
        mov 	saved_lc = ar.lc	// save the loop counter
	.body
	or	tmp3 = in0, in1 ;;	// tmp3 = dest | src
	or	tmp3 = tmp3, in2	// tmp3 = dest | src | len
	mov 	dest = in0		// dest
	mov 	src = in1		// src
	mov	len = in2		// len
	sub	tmp2 = r0, in0		// tmp2 = -dest
	cmp.eq	p6, p0 = in2, r0	// if (len == 0)
(p6)	br.cond.spnt .restore_and_exit;;// 	return dest;
	and	tmp4 = 7, tmp3 		// tmp4 = (dest | src | len) & 7
	cmp.le	p6, p0 = dest, src	// if dest <= src it's always safe
(p6)	br.cond.spnt .forward		// to copy forward
	add	tmp3 = src, len;;
	cmp.lt	p6, p0 = dest, tmp3	// if dest > src && dest < src + len
(p6)	br.cond.spnt .backward		// we have to copy backward

.forward:
	shr.u	loopcnt = len, 4 ;;	// loopcnt = len / 16
	cmp.ne	p6, p0 = tmp4, r0	// if ((dest | src | len) & 7 != 0)
(p6)	br.cond.sptk .next		//	goto next;

// The optimal case, when dest, src and len are all multiples of 8

	and	tmp3 = 0xf, len
	mov	pr.rot = 1 << 16	// set rotating predicates
	mov	ar.ec = MEMLAT + 1 ;;	// set the epilog counter
	cmp.ne	p6, p0 = tmp3, r0	// do we have to copy an extra word?
	adds	loopcnt = -1, loopcnt;;	// --loopcnt
(p6)	ld8	value = [src], 8;;
(p6)	st8	[dest] = value, 8	// copy the "odd" word
	mov	ar.lc = loopcnt 	// set the loop counter
	cmp.eq	p6, p0 = 8, len
(p6)	br.cond.spnt .restore_and_exit;;// the one-word special case
	adds	adest = 8, dest		// set adest one word ahead of dest
	adds	asrc = 8, src ;;	// set asrc one word ahead of src
	nop.b	0			// get the "golden" alignment for
	nop.b	0			// the next loop
.l0:
(p[0])		ld8	r[0] = [src], 16
(p[0])		ld8	q[0] = [asrc], 16
(p[MEMLAT])	st8	[dest] = r[MEMLAT], 16
(p[MEMLAT])	st8	[adest] = q[MEMLAT], 16
		br.ctop.dptk .l0 ;;

	mov	pr = saved_pr, -1	// restore the predicate registers
	mov	ar.lc = saved_lc	// restore the loop counter
	br.ret.sptk.many b0
.next:
	cmp.ge	p6, p0 = OP_T_THRES, len	// is len <= OP_T_THRES
	and	loopcnt = 7, tmp2 		// loopcnt = -dest % 8
(p6)	br.cond.spnt	.cpyfew			// copy byte by byte
	;;
	cmp.eq	p6, p0 = loopcnt, r0
(p6)	br.cond.sptk	.dest_aligned
	sub	len = len, loopcnt	// len -= -dest % 8
	adds	loopcnt = -1, loopcnt	// --loopcnt
	;;
	mov	ar.lc = loopcnt
.l1:					// copy -dest % 8 bytes
	ld1	value = [src], 1	// value = *src++
	;;
	st1	[dest] = value, 1	// *dest++ = value
	br.cloop.dptk .l1
.dest_aligned:
	and	sh1 = 7, src 		// sh1 = src % 8
	and	tmp2 = -8, len   	// tmp2 = len & -OPSIZ
	and	asrc = -8, src		// asrc = src & -OPSIZ  -- align src
	shr.u	loopcnt = len, 3	// loopcnt = len / 8
	and	len = 7, len;;		// len = len % 8
	adds	loopcnt = -1, loopcnt	// --loopcnt
	addl	tmp4 = @ltoff(.table), gp
	addl	tmp3 = @ltoff(.loop56), gp
	mov     ar.ec = MEMLAT + 1	// set EC
	mov     pr.rot = 1 << 16;;	// set rotating predicates
	mov	ar.lc = loopcnt		// set LC
	cmp.eq  p6, p0 = sh1, r0 	// is the src aligned?
(p6)    br.cond.sptk .src_aligned
	add	src = src, tmp2		// src += len & -OPSIZ
	shl	sh1 = sh1, 3		// sh1 = 8 * (src % 8)
	ld8	ploop56 = [tmp3]	// ploop56 = &loop56
	ld8	ptable = [tmp4];;	// ptable = &table
	add	tmp3 = ptable, sh1;;	// tmp3 = &table + sh1
	mov	ar.ec = MEMLAT + 1 + 1 // one more pass needed
	ld8	tmp4 = [tmp3];;		// tmp4 = loop offset
	sub	loopaddr = ploop56,tmp4	// loopadd = &loop56 - loop offset
	ld8	r[1] = [asrc], 8;;	// w0
	mov	b6 = loopaddr;;
	br	b6			// jump to the appropriate loop

	LOOP(8)
	LOOP(16)
	LOOP(24)
	LOOP(32)
	LOOP(40)
	LOOP(48)
	LOOP(56)

.src_aligned:
.l3:
(p[0])		ld8	r[0] = [src], 8
(p[MEMLAT])	st8	[dest] = r[MEMLAT], 8
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
	br.cloop.dptk	.l4 ;;
.restore_and_exit:
	mov     pr = saved_pr, -1    	// restore the predicate registers
	mov 	ar.lc = saved_lc	// restore the loop counter
	br.ret.sptk.many b0

// In the case of a backward copy, optimise only the case when everything
// is a multiple of 8, otherwise copy byte by byte.  The backward copy is
// used only when the blocks are overlapping and dest > src.

.backward:
	shr.u	loopcnt = len, 3	// loopcnt = len / 8
	add	src = src, len		// src points one byte past the end
	add	dest = dest, len ;; 	// dest points one byte past the end
	mov	ar.ec = MEMLAT + 1	// set the epilog counter
	mov	pr.rot = 1 << 16	// set rotating predicates
	adds	loopcnt = -1, loopcnt	// --loopcnt
	cmp.ne	p6, p0 = tmp4, r0	// if ((dest | src | len) & 7 != 0)
(p6)	br.cond.sptk .bytecopy ;;	// copy byte by byte backward
	adds	src = -8, src		// src points to the last word
	adds	dest = -8, dest 	// dest points to the last word
	mov	ar.lc = loopcnt;;	// set the loop counter
.l5:
(p[0])		ld8	r[0] = [src], -8
(p[MEMLAT])	st8	[dest] = r[MEMLAT], -8
		br.ctop.dptk .l5
		br.cond.sptk .restore_and_exit
.bytecopy:
	adds	src = -1, src		// src points to the last byte
	adds	dest = -1, dest		// dest points to the last byte
	adds	loopcnt = -1, len;;	// loopcnt = len - 1
	mov	ar.lc = loopcnt;;	// set the loop counter
.l6:
(p[0])		ld1	r[0] = [src], -1
(p[MEMLAT])	st1	[dest] = r[MEMLAT], -1
		br.ctop.dptk .l6
		br.cond.sptk .restore_and_exit
END(memmove)

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

libc_hidden_builtin_def (memmove)
