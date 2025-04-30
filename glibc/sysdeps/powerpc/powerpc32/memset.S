/* Optimized memset implementation for PowerPC.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
#include <rtld-global-offsets.h>

/* void * [r3] memset (void *s [r3], int c [r4], size_t n [r5]));
   Returns 's'.

   The memset is done in four sizes: byte (8 bits), word (32 bits),
   32-byte blocks (256 bits) and cache line size (128, 256, 1024 bits).
   There is a special case for setting whole cache lines to 0, which
   takes advantage of the dcbz instruction.  */

	.section	".text"
EALIGN (memset, 5, 1)

#define rTMP	r0
#define rRTN	r3	/* initial value of 1st argument */
#define rMEMP0	r3	/* original value of 1st arg */
#define rCHR	r4	/* char to set in each byte */
#define rLEN	r5	/* length of region to set */
#define rMEMP	r6	/* address at which we are storing */
#define rALIGN	r7	/* number of bytes we are setting now (when aligning) */
#define rMEMP2	r8

#define rPOS32	r7	/* constant +32 for clearing with dcbz */
#define rNEG64	r8	/* constant -64 for clearing with dcbz */
#define rNEG32	r9	/* constant -32 for clearing with dcbz */

#define rGOT	r9	/* Address of the Global Offset Table.  */
#define rCLS	r8	/* Cache line size obtained from static.  */
#define rCLM	r9	/* Cache line size mask to check for cache alignment.  */

/* take care of case for size <= 4  */
	cmplwi	cr1, rLEN, 4
	andi.	rALIGN, rMEMP0, 3
	mr	rMEMP, rMEMP0
	ble-	cr1, L(small)
/* align to word boundary  */
	cmplwi	cr5, rLEN, 31
	rlwimi	rCHR, rCHR, 8, 16, 23
	beq+	L(aligned)	/* 8th instruction from .align */
	mtcrf	0x01, rMEMP0
	subfic	rALIGN, rALIGN, 4
	add	rMEMP, rMEMP, rALIGN
	sub	rLEN, rLEN, rALIGN
	bf+	31, L(g0)
	stb	rCHR, 0(rMEMP0)
	bt	30, L(aligned)
L(g0):	sth	rCHR, -2(rMEMP)	/* 16th instruction from .align */
/* take care of case for size < 31 */
L(aligned):
	mtcrf	0x01, rLEN
	rlwimi	rCHR, rCHR, 16, 0, 15
	ble	cr5, L(medium)
/* align to cache line boundary...  */
	andi.	rALIGN, rMEMP, 0x1C
	subfic	rALIGN, rALIGN, 0x20
	beq	L(caligned)
	mtcrf	0x01, rALIGN
	add	rMEMP, rMEMP, rALIGN
	sub	rLEN, rLEN, rALIGN
	cmplwi	cr1, rALIGN, 0x10
	mr	rMEMP2, rMEMP
	bf	28, L(a1)
	stw	rCHR, -4(rMEMP2)
	stwu	rCHR, -8(rMEMP2)
L(a1):	blt	cr1, L(a2)
	stw	rCHR, -4(rMEMP2) /* 32nd instruction from .align */
	stw	rCHR, -8(rMEMP2)
	stw	rCHR, -12(rMEMP2)
	stwu	rCHR, -16(rMEMP2)
L(a2):	bf	29, L(caligned)
	stw	rCHR, -4(rMEMP2)
/* now aligned to a cache line.  */
L(caligned):
	cmplwi	cr1, rCHR, 0
	clrrwi.	rALIGN, rLEN, 5
	mtcrf	0x01, rLEN	/* 40th instruction from .align */

/* Check if we can use the special case for clearing memory using dcbz.
   This requires that we know the correct cache line size for this
   processor.  Getting the cache line size may require establishing GOT
   addressability, so branch out of line to set this up.  */
	beq	cr1, L(checklinesize)

/* Store blocks of 32-bytes (256-bits) starting on a 32-byte boundary.
   Can't assume that rCHR is zero or that the cache line size is either
   32-bytes or even known.  */
L(nondcbz):
	srwi	rTMP, rALIGN, 5
	mtctr	rTMP
	beq	L(medium)	/* we may not actually get to do a full line */
	clrlwi.	rLEN, rLEN, 27
	add	rMEMP, rMEMP, rALIGN
	li	rNEG64, -0x40
	bdz	L(cloopdone)	/* 48th instruction from .align */

/* We can't use dcbz here as we don't know the cache line size.  We can
   use "data cache block touch for store", which is safe.  */
L(c3):	dcbtst	rNEG64, rMEMP
	stw	rCHR, -4(rMEMP)
	stw	rCHR, -8(rMEMP)
	stw	rCHR, -12(rMEMP)
	stw	rCHR, -16(rMEMP)
	nop			/* let 601 fetch last 4 instructions of loop */
	stw	rCHR, -20(rMEMP)
	stw	rCHR, -24(rMEMP) /* 56th instruction from .align */
	nop			/* let 601 fetch first 8 instructions of loop */
	stw	rCHR, -28(rMEMP)
	stwu	rCHR, -32(rMEMP)
	bdnz	L(c3)
L(cloopdone):
	stw	rCHR, -4(rMEMP)
	stw	rCHR, -8(rMEMP)
	stw	rCHR, -12(rMEMP)
	stw	rCHR, -16(rMEMP) /* 64th instruction from .align */
	stw	rCHR, -20(rMEMP)
	cmplwi	cr1, rLEN, 16
	stw	rCHR, -24(rMEMP)
	stw	rCHR, -28(rMEMP)
	stwu	rCHR, -32(rMEMP)
	beqlr
	add	rMEMP, rMEMP, rALIGN
	b	L(medium_tail2)	/* 72nd instruction from .align */

	.align	5
	nop
/* Clear cache lines of memory in 128-byte chunks.
   This code is optimized for processors with 32-byte cache lines.
   It is further optimized for the 601 processor, which requires
   some care in how the code is aligned in the i-cache.  */
L(zloopstart):
	clrlwi	rLEN, rLEN, 27
	mtcrf	0x02, rALIGN
	srwi.	rTMP, rALIGN, 7
	mtctr	rTMP
	li	rPOS32, 0x20
	li	rNEG64, -0x40
	cmplwi	cr1, rLEN, 16	/* 8 */
	bf	26, L(z0)
	dcbz	0, rMEMP
	addi	rMEMP, rMEMP, 0x20
L(z0):	li	rNEG32, -0x20
	bf	25, L(z1)
	dcbz	0, rMEMP
	dcbz	rPOS32, rMEMP
	addi	rMEMP, rMEMP, 0x40 /* 16 */
L(z1):	cmplwi	cr5, rLEN, 0
	beq	L(medium)
L(zloop):
	dcbz	0, rMEMP
	dcbz	rPOS32, rMEMP
	addi	rMEMP, rMEMP, 0x80
	dcbz	rNEG64, rMEMP
	dcbz	rNEG32, rMEMP
	bdnz	L(zloop)
	beqlr	cr5
	b	L(medium_tail2)

	.align	5
L(small):
/* Memset of 4 bytes or less.  */
	cmplwi	cr5, rLEN, 1
	cmplwi	cr1, rLEN, 3
	bltlr	cr5
	stb	rCHR, 0(rMEMP)
	beqlr	cr5
	nop
	stb	rCHR, 1(rMEMP)
	bltlr	cr1
	stb	rCHR, 2(rMEMP)
	beqlr	cr1
	nop
	stb	rCHR, 3(rMEMP)
	blr

/* Memset of 0-31 bytes.  */
	.align	5
L(medium):
	cmplwi	cr1, rLEN, 16
L(medium_tail2):
	add	rMEMP, rMEMP, rLEN
L(medium_tail):
	bt-	31, L(medium_31t)
	bt-	30, L(medium_30t)
L(medium_30f):
	bt-	29, L(medium_29t)
L(medium_29f):
	bge-	cr1, L(medium_27t)
	bflr-	28
	stw	rCHR, -4(rMEMP)	/* 8th instruction from .align */
	stw	rCHR, -8(rMEMP)
	blr

L(medium_31t):
	stbu	rCHR, -1(rMEMP)
	bf-	30, L(medium_30f)
L(medium_30t):
	sthu	rCHR, -2(rMEMP)
	bf-	29, L(medium_29f)
L(medium_29t):
	stwu	rCHR, -4(rMEMP)
	blt-	cr1, L(medium_27f) /* 16th instruction from .align */
L(medium_27t):
	stw	rCHR, -4(rMEMP)
	stw	rCHR, -8(rMEMP)
	stw	rCHR, -12(rMEMP)
	stwu	rCHR, -16(rMEMP)
L(medium_27f):
	bflr-	28
L(medium_28t):
	stw	rCHR, -4(rMEMP)
	stw	rCHR, -8(rMEMP)
	blr

L(checklinesize):
/* If the remaining length is less the 32 bytes then don't bother getting
   the cache line size.  */
	beq	L(medium)
#ifdef PIC
	mflr	rTMP
/* Establishes GOT addressability so we can load the cache line size
   from rtld_global_ro. This value was set from the aux vector during
   startup.  */
	SETUP_GOT_ACCESS(rGOT,got_label)
	addis	rGOT,rGOT,_GLOBAL_OFFSET_TABLE_-got_label@ha
	addi	rGOT,rGOT,_GLOBAL_OFFSET_TABLE_-got_label@l
	mtlr	rTMP
#endif
/* Load rtld_global_ro._dl_cache_line_size.  */
	__GLRO(rCLS, rGOT, _dl_cache_line_size,
	       RTLD_GLOBAL_RO_DL_CACHE_LINE_SIZE_OFFSET)

/* If the cache line size was not set then goto to L(nondcbz), which is
   safe for any cache line size.  */
	cmplwi	cr1,rCLS,0
	beq	cr1,L(nondcbz)

/* If the cache line size is 32 bytes then goto to L(zloopstart),
   which is coded specifically for 32-byte lines (and 601).  */
	cmplwi	cr1,rCLS,32
	beq	cr1,L(zloopstart)

/* Now we know the cache line size and it is not 32-bytes.  However
   we may not yet be aligned to the cache line and may have a partial
   line to fill.  Touch it 1st to fetch the cache line.  */
	dcbtst	0,rMEMP

	addi	rCLM,rCLS,-1
L(getCacheAligned):
	cmplwi	cr1,rLEN,32
	and.	rTMP,rCLM,rMEMP
	blt	cr1,L(handletail32)
	beq	L(cacheAligned)
/* We are not aligned to start of a cache line yet.  Store 32-byte
   of data and test again.  */
	addi	rMEMP,rMEMP,32
	addi	rLEN,rLEN,-32
	stw	rCHR,-32(rMEMP)
	stw	rCHR,-28(rMEMP)
	stw	rCHR,-24(rMEMP)
	stw	rCHR,-20(rMEMP)
	stw	rCHR,-16(rMEMP)
	stw	rCHR,-12(rMEMP)
	stw	rCHR,-8(rMEMP)
	stw	rCHR,-4(rMEMP)
	b	L(getCacheAligned)

/* Now we are aligned to the cache line and can use dcbz.  */
L(cacheAligned):
	cmplw	cr1,rLEN,rCLS
	blt	cr1,L(handletail32)
	dcbz	0,rMEMP
	subf	rLEN,rCLS,rLEN
	add	rMEMP,rMEMP,rCLS
	b	L(cacheAligned)

/* We are here because; the cache line size was set, it was not
   32-bytes, and the remainder (rLEN) is now less than the actual cache
   line size.  Set up the preconditions for L(nondcbz) and go there to
   store the remaining bytes.  */
L(handletail32):
	clrrwi.	rALIGN, rLEN, 5
	b	L(nondcbz)

END (memset)
libc_hidden_builtin_def (memset)
