/* Set a block of memory to some byte value.  For SUN4V M7.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#include <sysdep.h>

#ifndef XCC
# define XCC    xcc
#endif
	.register	%g2, #scratch
	.register	%g3, #scratch

/* The algorithm is as follows :
 *
 *	For small 7 or fewer bytes stores, bytes will be stored.
 *
 *	For less than 32 bytes stores, align the address on 4 byte boundary.
 *	Then store as many 4-byte chunks, followed by trailing bytes.
 *
 *	For sizes greater than 32 bytes, align the address on 8 byte boundary.
 *	if (count >= 64) {
 *		store 8-bytes chunks to align the address on 64 byte boundary
 *		if (value to be set is zero && count >= MIN_ZERO) {
 *			Using BIS stores, set the first long word of each
 *			64-byte cache line to zero which will also clear the
 *			other seven long words of the cache line.
 *		}
 *		else if (count >= MIN_LOOP) {
 *			Using BIS stores, set the first long word of each of
 *			ST_CHUNK cache lines (64 bytes each) before the main
 *			loop is entered.
 *			In the main loop, continue pre-setting the first long
 *			word of each cache line ST_CHUNK lines in advance while
 *			setting the other seven long words (56 bytes) of each
 *			cache line until fewer than ST_CHUNK*64 bytes remain.
 *			Then set the remaining seven long words of each cache
 *			line that has already had its first long word set.
 *		}
 *		store remaining data in 64-byte chunks until less than
 *		64 bytes remain.
 *	}
 *	Store as many 8-byte chunks, followed by trailing bytes.
 *
 *
 * BIS = Block Init Store
 *   Doing the advance store of the first element of the cache line
 *   initiates the displacement of a cache line while only using a single
 *   instruction in the pipeline. That avoids various pipeline delays,
 *   such as filling the miss buffer. The performance effect is
 *   similar to prefetching for normal stores.
 *   The special case for zero fills runs faster and uses fewer instruction
 *   cycles than the normal memset loop.
 *
 * We only use BIS for memset of greater than MIN_LOOP bytes because a sequence
 * BIS stores must be followed by a membar #StoreStore. The benefit of
 * the BIS store must be balanced against the cost of the membar operation.
 */

/*
 * ASI_STBI_P marks the cache line as "least recently used"
 * which means if many threads are active, it has a high chance
 * of being pushed out of the cache between the first initializing
 * store and the final stores.
 * Thus, we use ASI_STBIMRU_P which marks the cache line as
 * "most recently used" for all but the last store to the cache line.
 */

#define ASI_BLK_INIT_QUAD_LDD_P 0xe2
#define ASI_ST_BLK_INIT_MRU_P 0xf2

#define ASI_STBI_P	ASI_BLK_INIT_QUAD_LDD_P
#define ASI_STBIMRU_P	ASI_ST_BLK_INIT_MRU_P

#define ST_CHUNK	24   /* multiple of 4 due to loop unrolling */
#define MIN_LOOP	(ST_CHUNK)*64
#define MIN_ZERO	256

#define EX_ST(x)	x
#define EX_RETVAL(x)	x
#define STORE_ASI(src,addr)	stxa src, [addr] ASI_STBIMRU_P
#define STORE_INIT(src,addr)	stxa src, [addr] ASI_STBI_P

#if IS_IN (libc)

	.text
	.align		32

ENTRY(__bzero_niagara7)
	/* bzero (dst, size)  */
	mov	%o1, %o2
	mov	0, %o1
	/* fall through into memset code */
END(__bzero_niagara7)

ENTRY(__memset_niagara7)
	/* memset (src, c, size)  */
	mov	%o0, %o5		/* copy sp1 before using it  */
	cmp	%o2, 7			/* if small counts, just write bytes  */
	bleu,pn %XCC, .Lwrchar
	 and	%o1, 0xff, %o1		/* o1 is (char)c  */

	sll	%o1, 8, %o3
	or	%o1, %o3, %o1		/* now o1 has 2 bytes of c  */
	sll	%o1, 16, %o3
	cmp	%o2, 32
	blu,pn	%XCC, .Lwdalign
	 or	%o1, %o3, %o1		/* now o1 has 4 bytes of c  */

	sllx	%o1, 32, %o3
	or	%o1, %o3, %o1		/* now o1 has 8 bytes of c  */

.Ldbalign:
	andcc	%o5, 7, %o3		/* is sp1 aligned on a 8 byte bound?  */
	bz,pt	%XCC, .Lblkalign	/* already long word aligned  */
	 sub	%o3, 8, %o3		/* -(bytes till long word aligned)  */

	add	%o2, %o3, %o2		/* update o2 with new count  */
	/* Set -(%o3) bytes till sp1 long word aligned  */
1:	stb	%o1, [%o5]		/* there is at least 1 byte to set  */
	inccc	%o3			/* byte clearing loop   */
	bl,pt	%XCC, 1b
	 inc	%o5

	/* Now sp1 is long word aligned (sp1 is found in %o5) */
.Lblkalign:
	cmp	%o2, 64		/* check if there are 64 bytes to set  */
	blu,pn	%XCC, .Lwrshort
	 mov	%o2, %o3

	andcc	%o5, 63, %o3		/* is sp1 block aligned?  */
	bz,pt	%XCC, .Lblkwr		/* now block aligned  */
	 sub	%o3, 64, %o3		/* o3 is -(bytes till block aligned)  */
	add	%o2, %o3, %o2		/* o2 is the remainder  */

	/* Store -(%o3) bytes till dst is block (64 byte) aligned.  */
	/* Use long word stores.  */
	/* Recall that dst is already long word aligned  */
1:
	addcc	%o3, 8, %o3
	stx	%o1, [%o5]
	bl,pt	%XCC, 1b
	 add	%o5, 8, %o5

	/* Now sp1 is block aligned  */
.Lblkwr:
	andn	%o2, 63, %o4		/* calculate size of blocks in bytes  */
	brz,pn	%o1, .Lwrzero		/* special case if c == 0  */
	 and	%o2, 63, %o3		/* %o3 = bytes left after blk stores  */

	cmp	%o4, MIN_LOOP		/* check for enough bytes to set  */
	blu,pn	%XCC, .Lshort_set	/* to justify cost of membar   */
	 nop				/* must be > pre-cleared lines  */

	/* initial cache-clearing stores  */
	/* get store pipeline moving  */

/*	Primary memset loop for large memsets  */
.Lwr_loop:
	mov	ST_CHUNK, %g1
.Lwr_loop_start:
	subcc	%g1, 4, %g1
	EX_ST(STORE_ASI(%o1,%o5))
	add	%o5, 64, %o5
	EX_ST(STORE_ASI(%o1,%o5))
	add	%o5, 64, %o5
	EX_ST(STORE_ASI(%o1,%o5))
	add	%o5, 64, %o5
	EX_ST(STORE_ASI(%o1,%o5))
	bgu	%XCC, .Lwr_loop_start
	 add	%o5, 64, %o5

	sub	%o5, ST_CHUNK*64, %o5	/* reset %o5  */
	mov	ST_CHUNK, %g1
	sub	%o5, 8, %o5		/* adjust %o5 for ASI store  */

.Lwr_loop_rest:
	stx	%o1,[%o5+8+8]
	sub	%o4, 64, %o4
	stx	%o1,[%o5+16+8]
	subcc	%g1, 1, %g1
	stx	%o1,[%o5+24+8]
	stx	%o1,[%o5+32+8]
	stx	%o1,[%o5+40+8]
	add	%o5, 64, %o5
	stx	%o1,[%o5-8]
	bgu	%XCC, .Lwr_loop_rest
	 EX_ST(STORE_INIT(%o1,%o5))

	 add	%o5, 8, %o5		/* restore %o5 offset  */

	/* If more than ST_CHUNK*64 bytes remain to set, continue  */
	/* setting the first long word of each cache line in advance  */
	/* to keep the store pipeline moving.  */

	cmp	%o4, ST_CHUNK*64
	bge,pt	%XCC, .Lwr_loop_start
	 mov	ST_CHUNK, %g1

	brz,a,pn %o4, .Lasi_done
	 nop

	sub	%o5, 8, %o5		/* adjust %o5 for ASI store  */
.Lwr_loop_small:
	add	%o5, 8, %o5		/* adjust %o5 for ASI store  */
	EX_ST(STORE_ASI(%o1,%o5))
	stx	%o1,[%o5+8]
	stx	%o1,[%o5+16]
	stx	%o1,[%o5+24]
	stx	%o1,[%o5+32]
	subcc	%o4, 64, %o4
	stx	%o1,[%o5+40]
	add	%o5, 56, %o5
	stx	%o1,[%o5-8]
	bgu,pt	%XCC, .Lwr_loop_small
	 EX_ST(STORE_INIT(%o1,%o5))

	ba	.Lasi_done
	 add	%o5, 8, %o5		/* restore %o5 offset  */

/*	Special case loop for zero fill memsets  */
/*	For each 64 byte cache line, single STBI to first element  */
/*	clears line  */
.Lwrzero:
	cmp	%o4, MIN_ZERO		/* check if enough bytes to set  */
					/* to pay %asi + membar cost  */
	blu	%XCC, .Lshort_set
	 nop
	sub	%o4, 256, %o4

.Lwrzero_loop:
	mov	64, %g3
	EX_ST(STORE_INIT(%o1,%o5))
	subcc	%o4, 256, %o4
	EX_ST(STORE_INIT(%o1,%o5+%g3))
	add	%o5, 256, %o5
	sub	%g3, 192, %g3
	EX_ST(STORE_INIT(%o1,%o5+%g3))
	add %g3, 64, %g3
	bge,pt	%XCC, .Lwrzero_loop
	 EX_ST(STORE_INIT(%o1,%o5+%g3))
	add	%o4, 256, %o4

	brz,pn	%o4, .Lbsi_done
	 nop
.Lwrzero_small:
	EX_ST(STORE_INIT(%o1,%o5))
	subcc	%o4, 64, %o4
	bgu,pt	%XCC, .Lwrzero_small
	 add	%o5, 64, %o5

.Lasi_done:
.Lbsi_done:
	membar	#StoreStore		/* required by use of BSI  */

.Lshort_set:
	cmp	%o4, 64			/* check if 64 bytes to set  */
	blu	%XCC, 5f
	 nop
4:					/* set final blocks of 64 bytes  */
	stx	%o1, [%o5]
	stx	%o1, [%o5+8]
	stx	%o1, [%o5+16]
	stx	%o1, [%o5+24]
	subcc	%o4, 64, %o4
	stx	%o1, [%o5+32]
	stx	%o1, [%o5+40]
	add	%o5, 64, %o5
	stx	%o1, [%o5-16]
	bgu,pt	%XCC, 4b
	 stx	%o1, [%o5-8]

5:
	/* Set the remaining long words  */
.Lwrshort:
	subcc	%o3, 8, %o3		/* Can we store any long words?  */
	blu,pn	%XCC, .Lwrchars
	 and	%o2, 7, %o2		/* calc bytes left after long words  */
6:
	subcc	%o3, 8, %o3
	stx	%o1, [%o5]		/* store the long words  */
	bgeu,pt %XCC, 6b
	 add	%o5, 8, %o5

.Lwrchars:				/* check for extra chars  */
	brnz	%o2, .Lwrfin
	 nop
	retl
	 nop

.Lwdalign:
	andcc	%o5, 3, %o3		/* is sp1 aligned on a word boundary  */
	bz,pn	%XCC, .Lwrword
	 andn	%o2, 3, %o3		/* create word sized count in %o3  */

	dec	%o2			/* decrement count  */
	stb	%o1, [%o5]		/* clear a byte  */
	b	.Lwdalign
	 inc	%o5			/* next byte  */

.Lwrword:
	subcc	%o3, 4, %o3
	st	%o1, [%o5]		/* 4-byte writing loop  */
	bnz,pt	%XCC, .Lwrword
	 add	%o5, 4, %o5
	and	%o2, 3, %o2		/* leftover count, if any  */

.Lwrchar:
	/* Set the remaining bytes, if any  */
	brz	%o2, .Lexit
	 nop
.Lwrfin:
	deccc	%o2
	stb	%o1, [%o5]
	bgu,pt	%XCC, .Lwrfin
	 inc	%o5
.Lexit:
	retl				/* %o0 was preserved  */
	 nop
END(__memset_niagara7)
#endif
