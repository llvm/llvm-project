/* Optimized memcpy implementation for CELL BE PowerPC.
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
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

#ifndef MEMCPY
# define MEMCPY memcpy
#endif

#define PREFETCH_AHEAD 6	/* no cache lines SRC prefetching ahead  */
#define ZERO_AHEAD 4		/* no cache lines DST zeroing ahead  */

/* memcpy routine optimized for CELL-BE-PPC	v2.0
 *
 * The CELL PPC core has 1 integer unit and 1 load/store unit
 * CELL:
 * 1st level data cache = 32K
 * 2nd level data cache = 512K
 * 3rd level data cache = 0K
 * With 3.2 GHz clockrate the latency to 2nd level cache is >36 clocks,
 * latency to memory is >400 clocks
 * To improve copy performance we need to prefetch source data
 * far ahead to hide this latency
 * For best performance instruction forms ending in "." like "andi."
 * should be avoided as the are implemented in microcode on CELL.
 * The below code is loop unrolled for the CELL cache line of 128 bytes
 */

.align  7

ENTRY_TOCLESS (MEMCPY, 5)
	CALL_MCOUNT 3

	dcbt	0,r4		/* Prefetch ONE SRC cacheline  */
	cmpldi	cr1,r5,16	/* is size < 16 ?  */
	mr	r6,r3
	blt+	cr1,.Lshortcopy

.Lbigcopy:
	neg	r8,r3		/* LS 3 bits = # bytes to 8-byte dest bdry  */
	clrldi  r8,r8,64-4	/* align to 16byte boundary  */
	sub     r7,r4,r3
	cmpldi	cr0,r8,0
	beq+	.Ldst_aligned

.Ldst_unaligned:
	mtcrf	0x01,r8		/* put #bytes to boundary into cr7  */
	subf	r5,r8,r5

	bf	cr7*4+3,1f
	lbzx	r0,r7,r6	/* copy 1 byte  */
	stb	r0,0(r6)
	addi	r6,r6,1
1:	bf	cr7*4+2,2f
	lhzx	r0,r7,r6	/* copy 2 byte  */
	sth	r0,0(r6)
	addi	r6,r6,2
2:	bf	cr7*4+1,4f
	lwzx	r0,r7,r6	/* copy 4 byte  */
	stw	r0,0(r6)
	addi	r6,r6,4
4:	bf	cr7*4+0,8f
	ldx	r0,r7,r6	/* copy 8 byte  */
	std	r0,0(r6)
	addi	r6,r6,8
8:
	add	r4,r7,r6

.Ldst_aligned:

	cmpdi	cr5,r5,128-1

	neg	r7,r6
	addi	r6,r6,-8	/* prepare for stdu  */
	addi	r4,r4,-8	/* prepare for ldu  */

	clrldi  r7,r7,64-7	/* align to cacheline boundary  */
	ble+	cr5,.Llessthancacheline

	cmpldi	cr6,r7,0
	subf	r5,r7,r5
	srdi	r7,r7,4		/* divide size by 16  */
	srdi	r10,r5,7	/* number of cache lines to copy  */

	cmpldi	r10,0
	li	r11,0		/* number cachelines to copy with prefetch  */
	beq	.Lnocacheprefetch

	cmpldi	r10,PREFETCH_AHEAD
	li	r12,128+8	/* prefetch distance  */
	ble	.Llessthanmaxprefetch

	subi	r11,r10,PREFETCH_AHEAD
	li	r10,PREFETCH_AHEAD

.Llessthanmaxprefetch:
	mtctr	r10

.LprefetchSRC:
	dcbt    r12,r4
	addi    r12,r12,128
	bdnz    .LprefetchSRC

.Lnocacheprefetch:
	mtctr	r7
	cmpldi	cr1,r5,128
	clrldi  r5,r5,64-7
	beq	cr6,.Lcachelinealigned

.Laligntocacheline:
	ld	r9,0x08(r4)
	ldu	r7,0x10(r4)
	std	r9,0x08(r6)
	stdu	r7,0x10(r6)
	bdnz	.Laligntocacheline


.Lcachelinealigned:		/* copy while cache lines  */

	blt-	cr1,.Llessthancacheline	/* size <128  */

.Louterloop:
	cmpdi   r11,0
	mtctr	r11
	beq-	.Lendloop

	li	r11,128*ZERO_AHEAD +8	/* DCBZ dist  */

.align	4
	/* Copy whole cachelines, optimized by prefetching SRC cacheline  */
.Lloop:				/* Copy aligned body  */
	dcbt	r12,r4		/* PREFETCH SOURCE some cache lines ahead  */
	ld	r9, 0x08(r4)
	dcbz	r11,r6
	ld	r7, 0x10(r4)	/* 4 register stride copy is optimal  */
	ld	r8, 0x18(r4)	/* to hide 1st level cache latency.  */
	ld	r0, 0x20(r4)
	std	r9, 0x08(r6)
	std	r7, 0x10(r6)
	std	r8, 0x18(r6)
	std	r0, 0x20(r6)
	ld	r9, 0x28(r4)
	ld	r7, 0x30(r4)
	ld	r8, 0x38(r4)
	ld	r0, 0x40(r4)
	std	r9, 0x28(r6)
	std	r7, 0x30(r6)
	std	r8, 0x38(r6)
	std	r0, 0x40(r6)
	ld	r9, 0x48(r4)
	ld	r7, 0x50(r4)
	ld	r8, 0x58(r4)
	ld	r0, 0x60(r4)
	std	r9, 0x48(r6)
	std	r7, 0x50(r6)
	std	r8, 0x58(r6)
	std	r0, 0x60(r6)
	ld	r9, 0x68(r4)
	ld	r7, 0x70(r4)
	ld	r8, 0x78(r4)
	ldu	r0, 0x80(r4)
	std	r9, 0x68(r6)
	std	r7, 0x70(r6)
	std	r8, 0x78(r6)
	stdu	r0, 0x80(r6)

	bdnz	.Lloop

.Lendloop:
	cmpdi	r10,0
	sldi	r10,r10,2	/* adjust from 128 to 32 byte stride  */
	beq-	.Lendloop2
	mtctr	r10

.Lloop2:			/* Copy aligned body  */
	ld	r9, 0x08(r4)
	ld	r7, 0x10(r4)
	ld	r8, 0x18(r4)
	ldu	r0, 0x20(r4)
	std	r9, 0x08(r6)
	std	r7, 0x10(r6)
	std	r8, 0x18(r6)
	stdu	r0, 0x20(r6)

	bdnz	.Lloop2
.Lendloop2:

.Llessthancacheline:		/* less than cache to do ?  */
	cmpldi	cr0,r5,16
	srdi	r7,r5,4		/* divide size by 16  */
	blt-	.Ldo_lt16
	mtctr	r7

.Lcopy_remaining:
	ld	r8,0x08(r4)
	ldu	r7,0x10(r4)
	std	r8,0x08(r6)
	stdu	r7,0x10(r6)
	bdnz	.Lcopy_remaining

.Ldo_lt16:			/* less than 16 ?  */
	cmpldi	cr0,r5,0	/* copy remaining bytes (0-15)  */
	beqlr+			/* no rest to copy  */
	addi	r4,r4,8
	addi	r6,r6,8

.Lshortcopy:			/* SIMPLE COPY to handle size =< 15 bytes  */
	mtcrf	0x01,r5
	sub	r7,r4,r6
	bf-	cr7*4+0,8f
	ldx	r0,r7,r6	/* copy 8 byte  */
	std	r0,0(r6)
	addi	r6,r6,8
8:
	bf	cr7*4+1,4f
	lwzx	r0,r7,r6	/* copy 4 byte  */
	stw	r0,0(r6)
	addi	r6,r6,4
4:
	bf	cr7*4+2,2f
	lhzx	r0,r7,r6	/* copy 2 byte  */
	sth	r0,0(r6)
	addi	r6,r6,2
2:
	bf	cr7*4+3,1f
	lbzx	r0,r7,r6	/* copy 1 byte  */
	stb	r0,0(r6)
1:	blr

END_GEN_TB (MEMCPY,TB_TOCLESS)
libc_hidden_builtin_def (memcpy)
