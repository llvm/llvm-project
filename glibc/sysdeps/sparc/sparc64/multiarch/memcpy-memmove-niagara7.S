/* Copy SIZE bytes from SRC to DEST.  For SUN4V M7.
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
	.register	%g2,#scratch
	.register	%g3,#scratch
	.register	%g6,#scratch

#define	FPRS_FEF	0x04

/*
 * ASI_STBI_P marks the cache line as "least recently used"
 * which means if many threads are active, it has a high chance
 * of being pushed out of the cache between the first initializing
 * store and the final stores.
 * Thus, in this algorithm we use ASI_STBIMRU_P which marks the
 * cache line as "most recently used" for all but the last cache
 * line.
 */

#define	ASI_BLK_INIT_QUAD_LDD_P	0xe2
#define	ASI_ST_BLK_INIT_MRU_P	0xf2

#define	ASI_STBI_P	ASI_BLK_INIT_QUAD_LDD_P
#define	ASI_STBIMRU_P	ASI_ST_BLK_INIT_MRU_P

#define	BLOCK_SIZE	64	/* L2 data cache line size  */
#define	SHORTCOPY	3
#define	SHORTCHECK	14
#define	SHORT_LONG	64	/* max copy for short longword-aligned case  */
				/* must be at least 64  */
#define	SMALL_MAX	255	/* max small copy for word/long aligned  */
#define	SMALL_UMAX	128	/* max small copy for unaligned case  */
#define	MED_WMAX	1023	/* max copy for medium word-aligned case  */
#define	MED_MAX		511	/* max copy for medium longword-aligned case  */
#define	ST_CHUNK	20	/* ST_CHUNK - block of values for BIS Store  */
/* on T4, prefetch 20 is a strong read prefetch to L1 and L2 data cache
 * prefetch 20 can cause inst pipeline to delay if data is in memory
 * prefetch 21 is a strong read prefetch to L2 data cache, not L1 data cache  */
#define	ALIGN_PRE	20	/* distance for aligned prefetch loop  */

#define EX_ST(x)	x
#define EX_RETVAL(x)	x
#define STORE_ASI(src,addr)	stxa src, [addr] ASI_STBIMRU_P
#define STORE_INIT(src,addr)	stxa src, [addr] ASI_STBI_P

#if IS_IN (libc)

	.text

ENTRY(__memmove_niagara7)
	/* %o0=dst, %o1=src, %o2=len */
	cmp	%o1, %o0	/* if from address is >= to use forward copy  */
	bgeu,pn	%XCC, .Lforcpy	/* else use backward if ...  */
	 sub	%o0, %o1, %o4	/* get difference of two addresses  */
	cmp	%o2, %o4	/* compare size and difference of addresses  */
	bleu,pn	%XCC, .Lforcpy	/* if size is bigger, do overlapped copy  */
	 add	%o1, %o2, %o5	/* get to end of source space  */

/* an overlapped copy that must be done "backwards"  */
.Lchksize:
	cmp	%o2, 8			/* less than 8 byte do byte copy  */
	blu,pn %XCC, 2f			/* else continue  */

/* Now size is bigger than 8  */
.Ldbalign:
	 add	%o0, %o2, %g1		/* get to end of dest space  */
	andcc	%g1, 7, %o3		/* %o3 has cnt til dst 8 byte align  */
	bz,a,pn	%XCC, .Ldbbck		/* skip if dst is 8 byte aligned  */
	 andn	%o2, 7, %o3		/* force %o3 cnt to multiple of 8  */
	sub	%o2, %o3, %o2		/* update o2 with new count  */

1:	dec	%o5			/* decrement source  */
	ldub	[%o5], %g1		/* load one byte  */
	deccc	%o3			/* decrement count  */
	bgu,pt	%XCC, 1b		/* if not done keep copying  */
	 stb	%g1, [%o5+%o4]		/* store one byte into dest  */
	andncc	%o2, 7, %o3		/* force %o3 cnt to multiple of 8  */
	bz,pn	%XCC, 2f		/* if size < 8, move to byte copy  */

/* Now Destination is 8 byte aligned  */
.Ldbbck:
	 andcc	%o5, 7, %o0		/* %o0 has src offset  */
	bz,a,pn	%XCC, .Ldbcopybc	/* if src is aligned do fast memmove  */
	 sub	%o2, %o3, %o2		/* Residue bytes in %o2  */

.Lcpy_dbwdbc:				/* alignment of src is needed  */
	sub	%o2, 8, %o2		/* set size one loop ahead  */
	sll	%o0, 3, %g1		/* %g1 is left shift  */
	mov	64, %g5			/* init %g5 to be 64  */
	sub	%g5, %g1, %g5		/* %g5 rightshift = (64 - leftshift)  */
	sub	%o5, %o0, %o5		/* align the src at 8 bytes.  */
	add	%o4, %o0, %o4		/* increase diff between src & dst  */
	ldx	[%o5], %o1		/* load first 8 bytes  */
	srlx	%o1, %g5, %o1
1:	sub	%o5, 8, %o5		/* subtract 8 from src  */
	ldx	[%o5], %o0		/* load 8 byte  */
	sllx	%o0, %g1, %o3		/* shift loaded val left to tmp reg  */
	or	%o1, %o3, %o3		/* align data  */
	stx	%o3, [%o5+%o4]		/* store 8 byte  */
	subcc	%o2, 8, %o2		/* subtract 8 byte from size  */
	bg,pt	%XCC, 1b		/* if size > 0 continue  */
	 srlx	%o0, %g5, %o1		/* move extra byte for the next use  */

	srl	%g1, 3, %o0		/* restore %o0 value for alignment  */
	add	%o5, %o0, %o5		/* restore src alignment  */
	sub	%o4, %o0, %o4		/* restore diff between src & dest  */

	ba	2f			/* branch to the trailing byte copy  */
	 add	%o2, 8, %o2		/* restore size value  */

.Ldbcopybc:				/* alignment of src is not needed  */
1:	sub	%o5, 8, %o5		/* subtract from src  */
	ldx	[%o5], %g1		/* load 8 bytes  */
	subcc	%o3, 8, %o3		/* subtract from size  */
	bgu,pt	%XCC, 1b		/* if size is bigger 0 continue  */
	 stx	%g1, [%o5+%o4]		/* store 8 bytes to destination  */

	ba	2f
	 nop

.Lbcbyte:
1:	ldub	[%o5], %g1		/* load one byte  */
	stb	%g1, [%o5+%o4]		/* store one byte  */
2:	deccc	%o2			/* decrement size  */
	bgeu,a,pt %XCC, 1b		/* if size is >= 0 continue  */
	 dec	%o5			/* decrement from address  */

.Lexitbc:				/* exit from backward copy  */
	retl
	 add	%o5, %o4, %o0		/* restore dest addr  */


/* Check to see if memmove is large aligned copy
 * If so, use special version of copy that avoids
 * use of block store init.  */
.Lforcpy:
	cmp	%o2, SMALL_MAX		/* check for not small case  */
	blt,pn	%XCC, .Lmv_short	/* merge with memcpy  */
	 mov	%o0, %g1		/* save %o0  */
	neg	%o0, %o5
	andcc	%o5, 7, %o5		/* bytes till DST 8 byte aligned  */
	brz,pt	%o5, .Lmv_dst_aligned_on_8

/* %o5 has the bytes to be written in partial store.  */
	 sub	%o2, %o5, %o2
	sub	%o1, %o0, %o1		/* %o1 gets the difference  */
7:					/* dst aligning loop  */
	ldub	[%o1+%o0], %o4		/* load one byte  */
	subcc	%o5, 1, %o5
	stb	%o4, [%o0]
	bgu,pt	%XCC, 7b
	 add	%o0, 1, %o0		/* advance dst  */
	add	%o1, %o0, %o1		/* restore %o1  */
.Lmv_dst_aligned_on_8:
	andcc	%o1, 7, %o5
	brnz,pn	%o5, .Lsrc_dst_unaligned_on_8
	 prefetch [%o1 + (1 * BLOCK_SIZE)], 20

.Lmv_src_dst_aligned_on_8:
/* check if we are copying MED_MAX or more bytes  */
	cmp	%o2, MED_MAX		/* limit to store buffer size  */
	bleu,pt	%XCC, .Lmedlong
	 prefetch [%o1 + (2 * BLOCK_SIZE)], 20

/* The mv_align loop below mimics the memcpy code for large aligned copies,
 * but does not use the ASI_STBI_P (block initializing store) performance
 * optimization.  This is used when memcpy is incorrectly invoked with
 * overlapping buffers.  */

.Lmv_large_align8_copy:			/* Src and dst share 8 byte align  */
					/* align dst to 64 byte boundary  */
	andcc	%o0, 0x3f, %o3		/* check for dst 64 byte aligned  */
	brz,pn	%o3, .Lmv_aligned_on_64
	 sub	%o3, 64, %o3		/* %o3 has negative bytes to move  */
	add	%o2, %o3, %o2		/* adjust remaining count  */
.Lmv_align_to_64:
	ldx	[%o1], %o4
	add	%o1, 8, %o1		/* increment src ptr  */
	addcc	%o3, 8, %o3
	stx	%o4, [%o0]
	brnz,pt	%o3, .Lmv_align_to_64
	 add	%o0, 8, %o0		/* increment dst ptr  */

.Lmv_aligned_on_64:
	andn	%o2, 0x3f, %o5		/* %o5 is multiple of block size  */
	and	%o2, 0x3f, %o2		/* residue bytes in %o2  */
.Lmv_align_loop:
	ldx	[%o1],%o4
	stx	%o4,[%o0]
	prefetch [%o0 + (10 * BLOCK_SIZE)], 22
	prefetch [%o1 + (10 * BLOCK_SIZE)], 21
	subcc	%o5, 64, %o5
	ldx	[%o1+8],%o4
	stx	%o4,[%o0+8]
	ldx	[%o1+16],%o4
	stx	%o4,[%o0+16]
	ldx	[%o1+24],%o4
	stx	%o4,[%o0+24]
	ldx	[%o1+32],%o4
	stx	%o4,[%o0+32]
	ldx	[%o1+40],%o4
	stx	%o4,[%o0+40]
	ldx	[%o1+48],%o4
	add	%o1, 64, %o1
	stx	%o4,[%o0+48]
	add	%o0, 64, %o0
	ldx	[%o1-8],%o4
	bgt,pt	%XCC, .Lmv_align_loop
	 stx	%o4,[%o0-8]

	ba	.Lmedlong
	 nop
END(__memmove_niagara7)

ENTRY(__mempcpy_niagara7)
	/* %o0=dst, %o1=src, %o2=len */
	ba,pt	%icc, 101f
	 add	%o0, %o2, %g1		/* save dst + len  */
END(__mempcpy_niagara7)

	.align	32
ENTRY(__memcpy_niagara7)
100:	/* %o0=dst, %o1=src, %o2=len */
	mov	%o0, %g1		/* save %o0  */
101:
#ifndef __arch64__
	srl	%o2, 0, %o2
#endif
	cmp	%o2, SMALL_MAX		/* check for not small case  */
	bgeu,pn	%XCC, .Lmedium		/* go to larger cases  */
.Lmv_short:
	 cmp	%o2, SHORTCOPY		/* check for really short case  */
	ble,pn	%XCC, .Lsmallfin
	 or	%o0, %o1, %o4		/* prepare alignment check  */
	andcc	%o4, 0x3, %o5		/* test for word alignment  */
	bnz,pn	%XCC, .Lsmallunalign	/* branch to non-word aligned case  */
	 nop
	subcc	%o2, 7, %o2		/* adjust count  */
	ble,pn	%XCC, .Lsmallwordx
	 andcc	%o4, 0x7, %o5		/* test for long alignment  */
/* 8 or more bytes, src and dest start on word boundary
 * %o4 contains or %o0, %o1  */
.Lsmalllong:
	bnz,pn	%XCC, .Lsmallwords	/* branch to word aligned case  */
	 cmp	%o2, SHORT_LONG-7
	bge,a	%XCC, .Lmedl64		/* if we branch  */
	 sub	%o2,56,%o2		/* adjust %o2 to -63 off count  */

/* slightly unroll the small_long_loop to improve very short copies  */
	cmp	%o2, 32-7
	blt,a,pn %XCC, .Lsmall_long_l
	 sub	%o1, %o0, %o1		/* %o1 gets the difference  */

	ldx	[%o1], %o5
	ldx	[%o1+8], %o4
	ldx	[%o1+16], %o3

	subcc	%o2, 24, %o2
	sub	%o1, %o0, %o1		/* %o1 gets the difference  */

	stx	%o5, [%o0]		/* write word  */
	stx	%o4, [%o0+8]		/* write word  */
	stx	%o3, [%o0+16]		/* write word  */

	add	%o0, 24, %o0

/* end loop unroll  */

.Lsmall_long_l:
	ldx	[%o1+%o0], %o3
	subcc	%o2, 8, %o2
	add	%o0, 8, %o0
	bgu,pn	%XCC, .Lsmall_long_l	/* loop until done  */
	 stx	%o3, [%o0-8]		/* write word  */
	addcc	%o2, 7, %o2		/* restore %o2 to correct count  */
	bnz,pn	%XCC, .Lsmall_long_x	/* check for completion  */
	 add	%o1, %o0, %o1		/* restore %o1  */
	retl
	 mov	EX_RETVAL(%g1), %o0	/* restore %o0  */
.Lsmall_long_x:
	cmp	%o2, 4			/* check for 4 or more bytes left  */
	blt,pn	%XCC, .Lsmallleft3	/* if not, go to finish up  */
	 nop
	lduw	[%o1], %o3
	add	%o1, 4, %o1
	subcc	%o2, 4, %o2
	stw	%o3, [%o0]
	bnz,pn	%XCC, .Lsmallleft3
	 add	%o0, 4, %o0
	retl
	 mov	EX_RETVAL(%g1), %o0	/* restore %o0  */

	.align 32
/* src and dest start on word boundary; 7 or fewer bytes  */
.Lsmallwordx:
	lduw	[%o1], %o3		/* read word  */
	addcc	%o2, 3, %o2		/* restore count  */
	bz,pt	%XCC, .Lsmallexit
	 stw	%o3, [%o0]		/* write word  */
	deccc	%o2			/* reduce count for cc test  */
	ldub	[%o1+4], %o3		/* load one byte  */
	bz,pt	%XCC, .Lsmallexit
	 stb	%o3, [%o0+4]		/* store one byte  */
	ldub	[%o1+5], %o3		/* load second byte  */
	deccc	%o2
	bz,pt	%XCC, .Lsmallexit
	 stb	%o3, [%o0+5]		/* store second byte  */
	ldub	[%o1+6], %o3		/* load third byte  */
	stb	%o3, [%o0+6]		/* store third byte  */
.Lsmallexit:
	retl
	 mov	EX_RETVAL(%g1), %o0	/* restore %o0  */

	.align 32
.Lsmallunalign:
	cmp	%o2, SHORTCHECK
	ble,pn	%XCC, .Lsmallrest
	 cmp	%o2, SMALL_UMAX
	bge,pt	%XCC, .Lmedium_join
	 andcc	%o1, 0x3, %o5		/* is src word aligned  */
	bz,pn	%XCC, .Laldst
	 cmp	%o5, 2			/* is src half-word aligned  */
	be,pt	%XCC, .Ls2algn
	 cmp	%o5, 3			/* src is byte aligned  */
.Ls1algn:
	ldub	[%o1], %o3		/* move 1 or 3 bytes to align it  */
	inc	1, %o1
	stb	%o3, [%o0]		/* move a byte to align src  */
	inc	1, %o0
	bne,pt	%XCC, .Ls2algn
	 dec	%o2
	b	.Lald			/* now go align dest  */
	 andcc	%o0, 0x3, %o5

.Ls2algn:
	lduh	[%o1], %o3		/* know src is 2 byte aligned  */
	inc	2, %o1
	srl	%o3, 8, %o4
	stb	%o4, [%o0]		/* have to do bytes,  */
	stb	%o3, [%o0 + 1]		/* do not know dst alignment  */
	inc	2, %o0
	dec	2, %o2

.Laldst:
	andcc	%o0, 0x3, %o5		/* align the destination address  */
.Lald:
	bz,pn	%XCC, .Lw4cp
	 cmp	%o5, 2
	be,pn	%XCC, .Lw2cp
	 cmp	%o5, 3
.Lw3cp:	lduw	[%o1], %o4
	inc	4, %o1
	srl	%o4, 24, %o5
	stb	%o5, [%o0]
	bne,pt	%XCC, .Lw1cp
	 inc	%o0
	dec	1, %o2
	andn	%o2, 3, %o3		/* %o3 is aligned word count  */
	dec	4, %o3			/* avoid reading beyond tail of src  */
	sub	%o1, %o0, %o1		/*  %o1 gets the difference  */

1:	sll	%o4, 8, %g5		/* save residual bytes  */
	lduw	[%o1+%o0], %o4
	deccc	4, %o3
	srl	%o4, 24, %o5		/* merge with residual  */
	or	%o5, %g5, %g5
	st	%g5, [%o0]
	bnz,pt	%XCC, 1b
	 inc	4, %o0
	sub	%o1, 3, %o1		/* used one byte of last word read  */
	and	%o2, 3, %o2
	b	7f
	 inc	4, %o2

.Lw1cp:	srl	%o4, 8, %o5
	sth	%o5, [%o0]
	inc	2, %o0
	dec	3, %o2
	andn	%o2, 3, %o3		/* %o3 is aligned word count  */
	dec	4, %o3			/* avoid reading beyond tail of src  */
	sub	%o1, %o0, %o1		/* %o1 gets the difference  */

2:	sll	%o4, 24, %g5		/* save residual bytes  */
	lduw	[%o1+%o0], %o4
	deccc	4, %o3
	srl	%o4, 8, %o5		/* merge with residual  */
	or	%o5, %g5, %g5
	st	%g5, [%o0]
	bnz,pt	%XCC, 2b
	 inc	4, %o0
	sub	%o1, 1, %o1		/* used 3 bytes of last word read  */
	and	%o2, 3, %o2
	b	7f
	 inc	4, %o2

.Lw2cp:	lduw	[%o1], %o4
	inc	4, %o1
	srl	%o4, 16, %o5
	sth	%o5, [%o0]
	inc	2, %o0
	dec	2, %o2
	andn	%o2, 3, %o3		/* %o3 is aligned word count  */
	dec	4, %o3			/* avoid reading beyond tail of src  */
	sub	%o1, %o0, %o1		/* %o1 gets the difference  */

3:	sll	%o4, 16, %g5		/* save residual bytes  */
	lduw	[%o1+%o0], %o4
	deccc	4, %o3
	srl	%o4, 16, %o5		/* merge with residual  */
	or	%o5, %g5, %g5
	st	%g5, [%o0]
	bnz,pt	%XCC, 3b
	 inc	4, %o0
	sub	%o1, 2, %o1		/* used two bytes of last word read  */
	and	%o2, 3, %o2
	b	7f
	 inc	4, %o2

.Lw4cp:	andn	%o2, 3, %o3		/* %o3 is aligned word count  */
	sub	%o1, %o0, %o1		/* %o1 gets the difference  */

1:	lduw	[%o1+%o0], %o4		/* read from address  */
	deccc	4, %o3			/* decrement count  */
	st	%o4, [%o0]		/* write at destination address  */
	bgu,pt	%XCC, 1b
	 inc	4, %o0			/* increment to address  */
	and	%o2, 3, %o2		/* number of leftover bytes, if any  */

	/* simple finish up byte copy, works with any alignment  */
7:
	add	%o1, %o0, %o1		/* restore %o1  */
.Lsmallrest:
	tst	%o2
	bz,pt	%XCC, .Lsmallx
	 cmp	%o2, 4
	blt,pn	%XCC, .Lsmallleft3
	 nop
	sub	%o2, 3, %o2
.Lsmallnotalign4:
	ldub	[%o1], %o3		/* read byte  */
	subcc	%o2, 4, %o2		/* reduce count by 4  */
	stb	%o3, [%o0]		/* write byte  */
	ldub	[%o1+1], %o3		/* repeat for total of 4 bytes  */
	add	%o1, 4, %o1		/* advance SRC by 4  */
	stb	%o3, [%o0+1]
	ldub	[%o1-2], %o3
	add	%o0, 4, %o0		/* advance DST by 4  */
	stb	%o3, [%o0-2]
	ldub	[%o1-1], %o3
	bgu,pt	%XCC, .Lsmallnotalign4	/* loop til 3 or fewer bytes remain  */
	 stb	%o3, [%o0-1]
	addcc	%o2, 3, %o2		/* restore count  */
	bz,pt	%XCC, .Lsmallx
.Lsmallleft3:				/* 1, 2, or 3 bytes remain  */
	 subcc	%o2, 1, %o2
	ldub	[%o1], %o3		/* load one byte  */
	bz,pt	%XCC, .Lsmallx
	 stb	%o3, [%o0]		/* store one byte  */
	ldub	[%o1+1], %o3		/* load second byte  */
	subcc	%o2, 1, %o2
	bz,pt	%XCC, .Lsmallx
	 stb	%o3, [%o0+1]		/* store second byte  */
	ldub	[%o1+2], %o3		/* load third byte  */
	stb	%o3, [%o0+2]		/* store third byte  */
.Lsmallx:
	retl
	 mov	EX_RETVAL(%g1), %o0	/* restore %o0  */

.Lsmallfin:
	tst	%o2
	bnz,pn	%XCC, .Lsmallleft3
	 nop
	retl
	 mov	EX_RETVAL(%g1), %o0	/* restore %o0  */

	.align 16
.Lsmallwords:
	lduw	[%o1], %o3		/* read word  */
	subcc	%o2, 8, %o2		/* update count  */
	stw	%o3, [%o0]		/* write word  */
	add	%o1, 8, %o1		/* update SRC  */
	lduw	[%o1-4], %o3		/* read word  */
	add	%o0, 8, %o0		/* update DST  */
	bgu,pt	%XCC, .Lsmallwords	/* loop until done  */
	 stw	%o3, [%o0-4]		/* write word  */
	addcc	%o2, 7, %o2		/* restore count  */
	bz,pt	%XCC, .Lsmallexit	/* check for completion  */
	 cmp	%o2, 4			/* check for 4 or more bytes left  */
	blt,pt	%XCC, .Lsmallleft3	/* if not, go to finish up  */
	 nop
	lduw	[%o1], %o3
	add	%o1, 4, %o1
	subcc	%o2, 4, %o2
	add	%o0, 4, %o0
	bnz,pn	%XCC, .Lsmallleft3
	 stw	%o3, [%o0-4]
	retl
	 mov	EX_RETVAL(%g1), %o0	/* restore %o0  */

	.align 16
.Lmedium:
.Lmedium_join:
	neg	%o0, %o5
	andcc	%o5, 7, %o5		/* bytes till DST 8 byte aligned  */
	brz,pt	%o5, .Ldst_aligned_on_8

	/* %o5 has the bytes to be written in partial store.  */
	 sub	%o2, %o5, %o2
	sub	%o1, %o0, %o1		/* %o1 gets the difference  */
7:					/* dst aligning loop  */
	ldub	[%o1+%o0], %o4		/* load one byte  */
	subcc	%o5, 1, %o5
	stb	%o4, [%o0]
	bgu,pt	%XCC, 7b
	 add	%o0, 1, %o0		/* advance dst  */
	add	%o1, %o0, %o1		/* restore %o1  */
.Ldst_aligned_on_8:
	andcc	%o1, 7, %o5
	brnz,pt	%o5, .Lsrc_dst_unaligned_on_8
	 nop

.Lsrc_dst_aligned_on_8:
	/* check if we are copying MED_MAX or more bytes  */
	cmp	%o2, MED_MAX		/* limit to store buffer size  */
	bgu,pn	%XCC, .Llarge_align8_copy
	 nop
/*
 * Special case for handling when src and dest are both long word aligned
 * and total data to move is less than MED_MAX bytes
 */
.Lmedlong:
	subcc	%o2, 63, %o2		/* adjust length to allow cc test  */
	ble,pn	%XCC, .Lmedl63		/* skip big loop if < 64 bytes  */
	 nop
.Lmedl64:
	ldx	[%o1], %o4		/* load  */
	subcc	%o2, 64, %o2		/* decrement length count  */
	stx	%o4, [%o0]		/* and store  */
	ldx	[%o1+8], %o3		/* a block of 64 bytes  */
	stx	%o3, [%o0+8]
	ldx	[%o1+16], %o4
	stx	%o4, [%o0+16]
	ldx	[%o1+24], %o3
	stx	%o3, [%o0+24]
	ldx	[%o1+32], %o4		/* load  */
	stx	%o4, [%o0+32]		/* and store  */
	ldx	[%o1+40], %o3		/* a block of 64 bytes  */
	add	%o1, 64, %o1		/* increase src ptr by 64  */
	stx	%o3, [%o0+40]
	ldx	[%o1-16], %o4
	add	%o0, 64, %o0		/* increase dst ptr by 64  */
	stx	%o4, [%o0-16]
	ldx	[%o1-8], %o3
	bgu,pt	%XCC, .Lmedl64		/* repeat if at least 64 bytes left  */
	 stx	%o3, [%o0-8]
.Lmedl63:
	addcc	%o2, 32, %o2		/* adjust remaining count  */
	ble,pt	%XCC, .Lmedl31		/* to skip if 31 or fewer bytes left  */
	 nop
	ldx	[%o1], %o4		/* load  */
	sub	%o2, 32, %o2		/* decrement length count  */
	stx	%o4, [%o0]		/* and store  */
	ldx	[%o1+8], %o3		/* a block of 32 bytes  */
	add	%o1, 32, %o1		/* increase src ptr by 32  */
	stx	%o3, [%o0+8]
	ldx	[%o1-16], %o4
	add	%o0, 32, %o0		/* increase dst ptr by 32  */
	stx	%o4, [%o0-16]
	ldx	[%o1-8], %o3
	stx	%o3, [%o0-8]
.Lmedl31:
	addcc	%o2, 16, %o2		/* adjust remaining count  */
	ble,pt	%XCC, .Lmedl15		/* skip if 15 or fewer bytes left  */
	 nop
	ldx	[%o1], %o4		/* load and store 16 bytes  */
	add	%o1, 16, %o1		/* increase src ptr by 16  */
	stx	%o4, [%o0]
	sub	%o2, 16, %o2		/* decrease count by 16  */
	ldx	[%o1-8], %o3
	add	%o0, 16, %o0		/* increase dst ptr by 16  */
	stx	%o3, [%o0-8]
.Lmedl15:
	addcc	%o2, 15, %o2		/* restore count  */
	bz,pt	%XCC, .Lsmallexit	/* exit if finished  */
	 cmp	%o2, 8
	blt,pt	%XCC, .Lmedw7		/* skip if 7 or fewer bytes left  */
	 tst	%o2
	ldx	[%o1], %o4		/* load 8 bytes  */
	add	%o1, 8, %o1		/* increase src ptr by 8  */
	add	%o0, 8, %o0		/* increase dst ptr by 8  */
	subcc	%o2, 8, %o2		/* decrease count by 8  */
	bnz,pn	%XCC, .Lmedw7
	 stx	%o4, [%o0-8]		/* and store 8 bytes  */
	retl
	 mov	EX_RETVAL(%g1), %o0	/* restore %o0  */

	.align 16
.Lsrc_dst_unaligned_on_8:
	/* DST is 8-byte aligned, src is not  */
	andcc	%o1, 0x3, %o5		/* test word alignment  */
	bnz,pt	%XCC, .Lunalignsetup	/* branch if not word aligned  */
	 nop

/*
 * Handle all cases where src and dest are aligned on word
 * boundaries. Use unrolled loops for better performance.
 * This option wins over standard large data move when
 * source and destination is in cache for medium
 * to short data moves.
 */
	cmp %o2, MED_WMAX		/* limit to store buffer size  */
	bge,pt	%XCC, .Lunalignrejoin	/* otherwise rejoin main loop  */
	 nop

	subcc	%o2, 31, %o2		/* adjust length to allow cc test  */
					/* for end of loop  */
	ble,pt	%XCC, .Lmedw31		/* skip big loop if less than 16  */
.Lmedw32:
	 ld	[%o1], %o4		/* move a block of 32 bytes  */
	sllx	%o4, 32, %o5
	ld	[%o1+4], %o4
	or	%o4, %o5, %o5
	stx	%o5, [%o0]
	subcc	%o2, 32, %o2		/* decrement length count  */
	ld	[%o1+8], %o4
	sllx	%o4, 32, %o5
	ld	[%o1+12], %o4
	or	%o4, %o5, %o5
	stx	%o5, [%o0+8]
	add	%o1, 32, %o1		/* increase src ptr by 32  */
	ld	[%o1-16], %o4
	sllx	%o4, 32, %o5
	ld	[%o1-12], %o4
	or	%o4, %o5, %o5
	stx	%o5, [%o0+16]
	add	%o0, 32, %o0		/* increase dst ptr by 32  */
	ld	[%o1-8], %o4
	sllx	%o4, 32, %o5
	ld	[%o1-4], %o4
	or	%o4, %o5, %o5
	bgu,pt	%XCC, .Lmedw32		/* repeat if at least 32 bytes left  */
	 stx	%o5, [%o0-8]
.Lmedw31:
	addcc	%o2, 31, %o2		/* restore count  */
	bz,pt	%XCC, .Lsmallexit	/* exit if finished  */
	 cmp	%o2, 16
	blt,pt	%XCC, .Lmedw15
	 nop
	ld	[%o1], %o4		/* move a block of 16 bytes  */
	sllx	%o4, 32, %o5
	subcc	%o2, 16, %o2		/* decrement length count  */
	ld	[%o1+4], %o4
	or	%o4, %o5, %o5
	stx	%o5, [%o0]
	add	%o1, 16, %o1		/* increase src ptr by 16  */
	ld	[%o1-8], %o4
	add	%o0, 16, %o0		/* increase dst ptr by 16  */
	sllx	%o4, 32, %o5
	ld	[%o1-4], %o4
	or	%o4, %o5, %o5
	stx	%o5, [%o0-8]
.Lmedw15:
	bz,pt	%XCC, .Lsmallexit	/* exit if finished  */
	 cmp	%o2, 8
	blt,pn	%XCC, .Lmedw7		/* skip if 7 or fewer bytes left  */
	 tst	%o2
	ld	[%o1], %o4		/* load 4 bytes  */
	subcc	%o2, 8, %o2		/* decrease count by 8  */
	stw	%o4, [%o0]		/* and store 4 bytes  */
	add	%o1, 8, %o1		/* increase src ptr by 8  */
	ld	[%o1-4], %o3		/* load 4 bytes  */
	add	%o0, 8, %o0		/* increase dst ptr by 8  */
	stw	%o3, [%o0-4]		/* and store 4 bytes  */
	bz,pt	%XCC, .Lsmallexit	/* exit if finished  */
.Lmedw7:				/* count is ge 1, less than 8  */
	 cmp	%o2, 4			/* check for 4 bytes left  */
	blt,pn	%XCC, .Lsmallleft3	/* skip if 3 or fewer bytes left  */
	 nop
	ld	[%o1], %o4		/* load 4 bytes  */
	add	%o1, 4, %o1		/* increase src ptr by 4  */
	add	%o0, 4, %o0		/* increase dst ptr by 4  */
	subcc	%o2, 4, %o2		/* decrease count by 4  */
	bnz,pt	%XCC, .Lsmallleft3
	 stw	%o4, [%o0-4]		/* and store 4 bytes  */
	retl
	 mov	EX_RETVAL(%g1), %o0	/* restore %o0  */

	.align 16
.Llarge_align8_copy:			/* Src and dst 8 byte aligned  */
	/* align dst to 64 byte boundary  */
	andcc	%o0, 0x3f, %o3		/* check for dst 64 byte aligned  */
	brz,pn	%o3, .Laligned_to_64
	 andcc	%o0, 8, %o3		/* odd long words to move?  */
	brz,pt	%o3, .Laligned_to_16
	 nop
	ldx	[%o1], %o4
	sub	%o2, 8, %o2
	add	%o1, 8, %o1		/* increment src ptr  */
	add	%o0, 8, %o0		/* increment dst ptr  */
	stx	%o4, [%o0-8]
.Laligned_to_16:
	andcc	%o0, 16, %o3		/* pair of long words to move?  */
	brz,pt	%o3, .Laligned_to_32
	 nop
	ldx	[%o1], %o4
	sub	%o2, 16, %o2
	stx	%o4, [%o0]
	add	%o1, 16, %o1		/* increment src ptr  */
	ldx	[%o1-8], %o4
	add	%o0, 16, %o0		/* increment dst ptr  */
	stx	%o4, [%o0-8]
.Laligned_to_32:
	andcc	%o0, 32, %o3		/* four long words to move?  */
	brz,pt	%o3, .Laligned_to_64
	 nop
	ldx	[%o1], %o4
	sub	%o2, 32, %o2
	stx	%o4, [%o0]
	ldx	[%o1+8], %o4
	stx	%o4, [%o0+8]
	ldx	[%o1+16], %o4
	stx	%o4, [%o0+16]
	add	%o1, 32, %o1		/* increment src ptr  */
	ldx	[%o1-8], %o4
	add	%o0, 32, %o0		/* increment dst ptr  */
	stx	%o4, [%o0-8]
.Laligned_to_64:
/*	Following test is included to avoid issues where existing executables
 *	incorrectly call memcpy with overlapping src and dest instead of memmove
 *
 *	if ( (src ge dst) and (dst+len > src)) go to overlap case
 *	if ( (src lt dst) and (src+len > dst)) go to overlap case
 */
	cmp	%o1,%o0
	bge,pt	%XCC, 1f
	 nop
/*				src+len > dst?  */
	add	%o1, %o2, %o4
	cmp	%o4, %o0
	bgt,pt	%XCC, .Lmv_aligned_on_64
	 nop
	ba	2f
	 nop
1:
/*				dst+len > src?  */
	add	%o0, %o2, %o4
	cmp	%o4, %o1
	bgt,pt	%XCC, .Lmv_aligned_on_64
	 nop
2:
/*	handle non-overlapped copies
 *
 *	Using block init store (BIS) instructions to avoid fetching cache
 *	lines from memory. Use ST_CHUNK stores to first element of each cache
 *	line (similar to prefetching) to avoid overfilling STQ or miss buffers.
 *	Gives existing cache lines time to be moved out of L1/L2/L3 cache.
 */
	andn	%o2, 0x3f, %o5		/* %o5 is multiple of block size  */
	and	%o2, 0x3f, %o2		/* residue bytes in %o2  */

/*	We use ASI_STBIMRU_P for the first store to each cache line
 *	followed by ASI_STBI_P (mark as LRU) for the last store. That
 *	mixed approach reduces the chances the cache line is removed
 *	before we finish setting it, while minimizing the effects on
 *	other cached values during a large memcpy
 *
 *	Intermediate stores can be normal since first BIS activates the
 *	cache line in the L2 cache.
 *
 *	ST_CHUNK batches up initial BIS operations for several cache lines
 *	to allow multiple requests to not be blocked by overflowing the
 *	the store miss buffer. Then the matching stores for all those
 *	BIS operations are executed.
 */

.Lalign_loop:
	cmp	%o5, ST_CHUNK*64
	blu,pt	%XCC, .Lalign_short
	 mov	ST_CHUNK, %o3
	sllx	%o3, 6, %g5		/* ST_CHUNK*64  */

.Lalign_loop_start:
	prefetch [%o1 + (ALIGN_PRE * BLOCK_SIZE)], 21
	subcc	%o3, 2, %o3
	ldx	[%o1], %o4
	add	%o1, 128, %o1
	EX_ST(STORE_ASI(%o4, %o0))
	add	%o0, 64, %o0
	ldx	[%o1-64], %o4
	EX_ST(STORE_ASI(%o4, %o0))
	add	%o0, 64, %o0
	bgu,pt	%XCC, .Lalign_loop_start
	 prefetch [%o1 + ((ALIGN_PRE-1) * BLOCK_SIZE)], 21

	mov	ST_CHUNK, %o3
	sub	%o1, %g5, %o1		/* reset %o1  */
	sub	%o0, %g5, %o0		/* reset %o0  */

	sub	%o0, 8, %o0		/* adjust %o0 for ASI alignment  */
.Lalign_loop_rest:
	ldx	[%o1+8],%o4
	add	%o0, 64, %o0
	stx	%o4, [%o0-48]
	subcc	%o3, 1, %o3
	ldx	[%o1+16],%o4
	stx	%o4, [%o0-40]
	sub	%o5, 64, %o5
	ldx	[%o1+24],%o4
	stx	%o4, [%o0-32]
	ldx	[%o1+32],%o4
	stx	%o4, [%o0-24]
	ldx	[%o1+40],%o4
	stx	%o4, [%o0-16]
	ldx	[%o1+48],%o4
	stx	%o4, [%o0-8]
	add	%o1, 64, %o1
	ldx	[%o1-8],%o4
	bgu,pt	%XCC, .Lalign_loop_rest
	 EX_ST(STORE_INIT(%o4,%o0))	/* mark cache line as LRU  */

	mov	ST_CHUNK, %o3
	cmp	%o5, ST_CHUNK*64
	bgu,pt	%XCC, .Lalign_loop_start
	 add	%o0, 8, %o0		/* restore %o0 from ASI alignment  */

	cmp	%o5, 0
	beq,pt	%XCC, .Lalign_done

/* no prefetches needed in these loops
 * since we are within ALIGN_PRE of the end */
.Lalign_short:
	 srl	%o5, 6, %o3
.Lalign_loop_short:
	subcc	%o3, 1, %o3
	ldx	[%o1], %o4
	add	%o1, 64, %o1
	EX_ST(STORE_ASI(%o4, %o0))
	bgu,pt	%XCC, .Lalign_loop_short
	 add	%o0, 64, %o0

	sub	%o1, %o5, %o1		/* reset %o1  */
	sub	%o0, %o5, %o0		/* reset %o0  */

	sub	%o0, 8, %o0		/* adjust %o0 for ASI alignment  */
.Lalign_short_rest:
	ldx	[%o1+8],%o4
	add	%o0, 64, %o0
	stx	%o4, [%o0-48]
	ldx	[%o1+16],%o4
	subcc	%o5, 64, %o5
	stx	%o4, [%o0-40]
	ldx	[%o1+24],%o4
	stx	%o4, [%o0-32]
	ldx	[%o1+32],%o4
	stx	%o4, [%o0-24]
	ldx	[%o1+40],%o4
	stx	%o4, [%o0-16]
	ldx	[%o1+48],%o4
	stx	%o4, [%o0-8]
	add	%o1, 64, %o1
	ldx	[%o1-8],%o4
	bgu,pt	%XCC, .Lalign_short_rest
	 EX_ST(STORE_INIT(%o4,%o0))	/* mark cache line as LRU  */

	add	%o0, 8, %o0		/* restore %o0 from ASI alignment  */

.Lalign_done:
	cmp	%o2, 0
	membar	#StoreStore
	bne,pt	%XCC, .Lmedl63
	 subcc	%o2, 63, %o2		/* adjust length to allow cc test  */
	retl
	 mov	EX_RETVAL(%g1), %o0	/* restore %o0  */

	.align 16
	/* Dst is on 8 byte boundary; src is not; remaining cnt > SMALL_MAX  */
	/* Since block load/store and BIS are not in use for unaligned data,
	 * no need to align dst on 64 byte cache line boundary  */
.Lunalignsetup:
.Lunalignrejoin:
	rd	%fprs, %g5		/* check for unused fp  */
	/* if fprs.fef == 0, set it.
	 * Setting it when already set costs more than checking */
	andcc	%g5, FPRS_FEF, %g5	/* test FEF, fprs.du = fprs.dl = 0  */
	bz,a	%XCC, 1f
	 wr	%g0, FPRS_FEF, %fprs	/* fprs.fef = 1  */
1:
	andn	%o2, 0x3f, %o5		/* %o5 is multiple of block size  */
	and	%o2, 0x3f, %o2		/* residue bytes in %o2  */
	cmp	%o2, 8			/* Insure we do not load beyond  */
	bgt,pt	%XCC, .Lunalign_adjust	/* end of source buffer  */
	 andn	%o1, 0x7, %o4		/* %o4 has 8 byte aligned src addr  */
	add	%o2, 64, %o2		/* adjust to leave loop  */
	sub	%o5, 64, %o5		/* early if necessary  */
.Lunalign_adjust:
	alignaddr %o1, %g0, %g0		/* generate %gsr  */
	add	%o1, %o5, %o1		/* advance %o1 to after blocks  */
	ldd	[%o4], %f0
.Lunalign_loop:
	prefetch [%o0 + (9 * BLOCK_SIZE)], 20
	ldd	[%o4+8], %f2
	faligndata %f0, %f2, %f16
	ldd	[%o4+16], %f4
	subcc	%o5, BLOCK_SIZE, %o5
	std	%f16, [%o0]
	faligndata %f2, %f4, %f18
	ldd	[%o4+24], %f6
	std	%f18, [%o0+8]
	faligndata %f4, %f6, %f20
	ldd	[%o4+32], %f8
	std	%f20, [%o0+16]
	faligndata %f6, %f8, %f22
	ldd	[%o4+40], %f10
	std	%f22, [%o0+24]
	faligndata %f8, %f10, %f24
	ldd	[%o4+48], %f12
	std	%f24, [%o0+32]
	faligndata %f10, %f12, %f26
	ldd	[%o4+56], %f14
	add	%o4, BLOCK_SIZE, %o4
	std	%f26, [%o0+40]
	faligndata %f12, %f14, %f28
	ldd	[%o4], %f0
	std	%f28, [%o0+48]
	faligndata %f14, %f0, %f30
	std	%f30, [%o0+56]
	add	%o0, BLOCK_SIZE, %o0
	bgu,pt	%XCC, .Lunalign_loop
	 prefetch [%o4 + (11 * BLOCK_SIZE)], 20

	/* Handle trailing bytes, 64 to 127
	 * Dest long word aligned, Src not long word aligned  */
	cmp	%o2, 15
	bleu,pt	%XCC, .Lunalign_short

	 andn	%o2, 0x7, %o5		/* %o5 is multiple of 8  */
	and	%o2, 0x7, %o2		/* residue bytes in %o2  */
	add	%o2, 8, %o2
	sub	%o5, 8, %o5		/* do not load past end of src  */
	andn	%o1, 0x7, %o4		/* %o4 has 8 byte aligned src addr  */
	add	%o1, %o5, %o1		/* move %o1 to after multiple of 8  */
	ldd	[%o4], %f0		/* fetch partial word  */
.Lunalign_by8:
	ldd	[%o4+8], %f2
	add	%o4, 8, %o4
	faligndata %f0, %f2, %f16
	subcc	%o5, 8, %o5
	std	%f16, [%o0]
	fsrc2	%f2, %f0
	bgu,pt	%XCC, .Lunalign_by8
	 add	%o0, 8, %o0

.Lunalign_short:			/* restore fprs state */
	brnz,pt	%g5, .Lsmallrest
	 nop
	ba	.Lsmallrest
	 wr	%g5, %g0, %fprs
END(__memcpy_niagara7)

#endif
