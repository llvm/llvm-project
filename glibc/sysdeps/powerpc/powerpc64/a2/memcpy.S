/* Optimized memcpy implementation for PowerPC A2.
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
   Contributed by Michael Brutman <brutman@us.ibm.com>.
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

#ifndef MEMCPY
# define MEMCPY memcpy
#endif

#define PREFETCH_AHEAD 4        /* no cache lines SRC prefetching ahead  */
#define ZERO_AHEAD 2            /* no cache lines DST zeroing ahead  */

	.section        ".toc","aw"
__GLRO_DEF(dl_cache_line_size)


	.section        ".text"
	.align 2


	.machine  a2
ENTRY (MEMCPY, 5)
	CALL_MCOUNT 3

	dcbt    0,r4            /* Prefetch ONE SRC cacheline  */
	cmpldi  cr1,r5,16       /* is size < 16 ?  */
	mr      r6,r3           /* Copy dest reg to r6; */
	blt+    cr1,L(shortcopy)


	/* Big copy (16 bytes or more)

	   Figure out how far to the nearest quadword boundary, or if we are
	   on one already.  Also get the cache line size.

	   r3 - return value (always)
	   r4 - current source addr
	   r5 - copy length
	   r6 - current dest addr
	*/

	neg     r8,r3           /* LS 4 bits = # bytes to 8-byte dest bdry  */
	/* Get the cache line size.  */
	__GLRO (r9, dl_cache_line_size,
		RTLD_GLOBAL_RO_DL_CACHE_LINE_SIZE_OFFSET)
	clrldi  r8,r8,64-4      /* align to 16byte boundary  */
	sub     r7,r4,r3        /* compute offset to src from dest */
	cmpldi  cr0,r8,0        /* Were we aligned on a 16 byte bdy? */
	addi    r10,r9,-1       /* Cache line mask */
	beq+    L(dst_aligned)



	/* Destination is not aligned on quadword boundary.  Get us to one.

	   r3 - return value (always)
	   r4 - current source addr
	   r5 - copy length
	   r6 - current dest addr
	   r7 - offset to src from dest
	   r8 - number of bytes to quadword boundary
	*/

	mtcrf   0x01,r8         /* put #bytes to boundary into cr7  */
	subf    r5,r8,r5        /* adjust remaining len */

	bf      cr7*4+3,1f
	lbzx    r0,r7,r6        /* copy 1 byte addr */
	stb     r0,0(r6)
	addi    r6,r6,1
1:
	bf      cr7*4+2,2f
	lhzx    r0,r7,r6        /* copy 2 byte addr */
	sth     r0,0(r6)
	addi    r6,r6,2
2:
	bf      cr7*4+1,4f
	lwzx    r0,r7,r6        /* copy 4 byte addr */
	stw     r0,0(r6)
	addi    r6,r6,4
4:
	bf      cr7*4+0,8f
	ldx     r0,r7,r6        /* copy 8 byte addr */
	std     r0,0(r6)
	addi    r6,r6,8
8:
	add     r4,r7,r6        /* update src addr */



	/* Dest is quadword aligned now.

	   Lots of decisions to make.  If we are copying less than a cache
	   line we won't be here long.  If we are not on a cache line
	   boundary we need to get there.  And then we need to figure out
	   how many cache lines ahead to pre-touch.

	   r3 - return value (always)
	   r4 - current source addr
	   r5 - copy length
	   r6 - current dest addr
	*/


	.align 4
L(dst_aligned):
	cmpdi	cr0,r9,0	/* Cache line size set? */
	bne+	cr0,L(cachelineset)

/* Cache line size not set: generic byte copy without much optimization */
	clrldi.	r0,r5,63	/* If length is odd copy one byte */
	beq	L(cachelinenotset_align)
	lbz	r7,0(r4)	/* Read one byte from source */
	addi	r5,r5,-1	/* Update length */
	addi	r4,r4,1		/* Update source pointer address */
	stb	r7,0(r6)	/* Store one byte at dest */
	addi	r6,r6,1		/* Update dest pointer address */
L(cachelinenotset_align):
	cmpdi	cr7,r5,0	/* If length is 0 return */
	beqlr	cr7
	ori	r2,r2,0		/* Force a new dispatch group */
L(cachelinenotset_loop):
	addic.	r5,r5,-2	/* Update length */
	lbz	r7,0(r4)	/* Load 2 bytes from source */
	lbz	r8,1(r4)
	addi	r4,r4,2		/* Update source pointer address */
	stb	r7,0(r6)	/* Store 2 bytes on dest */
	stb	r8,1(r6)
	addi	r6,r6,2		/* Update dest pointer address */
	bne	L(cachelinenotset_loop)
	blr


L(cachelineset):
	cmpd	cr5,r5,r10       /* Less than a cacheline to go? */

	neg     r7,r6           /* How far to next cacheline bdy? */

	addi    r6,r6,-8        /* prepare for stdu  */
	cmpdi   cr0,r9,128
	addi    r4,r4,-8        /* prepare for ldu  */


	ble+    cr5,L(lessthancacheline)

	beq-    cr0,L(big_lines) /* 128 byte line code */



	/* More than a cacheline left to go, and using 64 byte cachelines */

	clrldi  r7,r7,64-6      /* How far to next cacheline bdy? */

	cmpldi  cr6,r7,0        /* Are we on a cacheline bdy already? */

	/* Reduce total len by what it takes to get to the next cache line */
	subf    r5,r7,r5
	srdi    r7,r7,4         /* How many qws to get to the line bdy? */

	/* How many full cache lines to copy after getting to a line bdy? */
	srdi    r10,r5,6

	cmpldi  r10,0           /* If no full cache lines to copy ... */
	li      r11,0           /* number cachelines to copy with prefetch  */
	beq     L(nocacheprefetch)


	/* We are here because we have at least one full cache line to copy,
	   and therefore some pre-touching to do. */

	cmpldi  r10,PREFETCH_AHEAD
	li      r12,64+8        /* prefetch distance  */
	ble     L(lessthanmaxprefetch)

	/* We can only do so much pre-fetching.  R11 will have the count of
	   lines left to prefetch after the initial batch of prefetches
	   are executed. */

	subi    r11,r10,PREFETCH_AHEAD
	li      r10,PREFETCH_AHEAD

L(lessthanmaxprefetch):
	mtctr   r10

	/* At this point r10/ctr hold the number of lines to prefetch in this
	   initial batch, and r11 holds any remainder. */

L(prefetchSRC):
	dcbt    r12,r4
	addi    r12,r12,64
	bdnz    L(prefetchSRC)


	/* Prefetching is done, or was not needed.

	   cr6 - are we on a cacheline boundary already?
	   r7  - number of quadwords to the next cacheline boundary
	*/

L(nocacheprefetch):
	mtctr   r7

	cmpldi  cr1,r5,64   /* Less than a cache line to copy? */

	/* How many bytes are left after we copy whatever full
	   cache lines we can get? */
	clrldi  r5,r5,64-6

	beq     cr6,L(cachelinealigned)


	/* Copy quadwords up to the next cacheline boundary */

L(aligntocacheline):
	ld      r9,0x08(r4)
	ld      r7,0x10(r4)
	addi    r4,r4,0x10
	std     r9,0x08(r6)
	stdu    r7,0x10(r6)
	bdnz    L(aligntocacheline)


	.align 4
L(cachelinealigned):            /* copy while cache lines  */

	blt-    cr1,L(lessthancacheline) /* size <64  */

L(outerloop):
	cmpdi   r11,0
	mtctr   r11
	beq-    L(endloop)

	li      r11,64*ZERO_AHEAD +8    /* DCBZ dist  */

	.align  4
	/* Copy whole cachelines, optimized by prefetching SRC cacheline  */
L(loop):                        /* Copy aligned body  */
	dcbt    r12,r4          /* PREFETCH SOURCE some cache lines ahead  */
	ld      r9, 0x08(r4)
	dcbz    r11,r6
	ld      r7, 0x10(r4)
	ld      r8, 0x18(r4)
	ld      r0, 0x20(r4)
	std     r9, 0x08(r6)
	std     r7, 0x10(r6)
	std     r8, 0x18(r6)
	std     r0, 0x20(r6)
	ld      r9, 0x28(r4)
	ld      r7, 0x30(r4)
	ld      r8, 0x38(r4)
	ld      r0, 0x40(r4)
	addi    r4, r4,0x40
	std     r9, 0x28(r6)
	std     r7, 0x30(r6)
	std     r8, 0x38(r6)
	stdu    r0, 0x40(r6)

	bdnz    L(loop)


L(endloop):
	cmpdi   r10,0
	beq-    L(endloop2)
	mtctr   r10

L(loop2):                       /* Copy aligned body  */
	ld      r9, 0x08(r4)
	ld      r7, 0x10(r4)
	ld      r8, 0x18(r4)
	ld      r0, 0x20(r4)
	std     r9, 0x08(r6)
	std     r7, 0x10(r6)
	std     r8, 0x18(r6)
	std     r0, 0x20(r6)
	ld      r9, 0x28(r4)
	ld      r7, 0x30(r4)
	ld      r8, 0x38(r4)
	ld      r0, 0x40(r4)
	addi    r4, r4,0x40
	std     r9, 0x28(r6)
	std     r7, 0x30(r6)
	std     r8, 0x38(r6)
	stdu    r0, 0x40(r6)

	bdnz    L(loop2)
L(endloop2):


	.align 4
L(lessthancacheline):           /* Was there less than cache to do ?  */
	cmpldi  cr0,r5,16
	srdi    r7,r5,4         /* divide size by 16  */
	blt-    L(do_lt16)
	mtctr   r7

L(copy_remaining):
	ld      r8,0x08(r4)
	ld      r7,0x10(r4)
	addi    r4,r4,0x10
	std     r8,0x08(r6)
	stdu    r7,0x10(r6)
	bdnz    L(copy_remaining)

L(do_lt16):                     /* less than 16 ?  */
	cmpldi  cr0,r5,0        /* copy remaining bytes (0-15)  */
	beqlr+                  /* no rest to copy  */
	addi    r4,r4,8
	addi    r6,r6,8

L(shortcopy):                   /* SIMPLE COPY to handle size =< 15 bytes  */
	mtcrf   0x01,r5
	sub     r7,r4,r6
	bf-     cr7*4+0,8f
	ldx     r0,r7,r6        /* copy 8 byte  */
	std     r0,0(r6)
	addi    r6,r6,8
8:
	bf      cr7*4+1,4f
	lwzx    r0,r7,r6        /* copy 4 byte  */
	stw     r0,0(r6)
	addi    r6,r6,4
4:
	bf      cr7*4+2,2f
	lhzx    r0,r7,r6        /* copy 2 byte  */
	sth     r0,0(r6)
	addi    r6,r6,2
2:
	bf      cr7*4+3,1f
	lbzx    r0,r7,r6        /* copy 1 byte  */
	stb     r0,0(r6)
1:
	blr





	/* Similar to above, but for use with 128 byte lines. */


L(big_lines):

	clrldi  r7,r7,64-7      /* How far to next cacheline bdy? */

	cmpldi  cr6,r7,0        /* Are we on a cacheline bdy already? */

	/* Reduce total len by what it takes to get to the next cache line */
	subf    r5,r7,r5
	srdi    r7,r7,4         /* How many qws to get to the line bdy? */

	/* How many full cache lines to copy after getting to a line bdy? */
	srdi    r10,r5,7

	cmpldi  r10,0           /* If no full cache lines to copy ... */
	li      r11,0           /* number cachelines to copy with prefetch  */
	beq     L(nocacheprefetch_128)


	/* We are here because we have at least one full cache line to copy,
	   and therefore some pre-touching to do. */

	cmpldi  r10,PREFETCH_AHEAD
	li      r12,128+8       /* prefetch distance  */
	ble     L(lessthanmaxprefetch_128)

	/* We can only do so much pre-fetching.  R11 will have the count of
	   lines left to prefetch after the initial batch of prefetches
	   are executed. */

	subi    r11,r10,PREFETCH_AHEAD
	li      r10,PREFETCH_AHEAD

L(lessthanmaxprefetch_128):
	mtctr   r10

	/* At this point r10/ctr hold the number of lines to prefetch in this
	   initial batch, and r11 holds any remainder. */

L(prefetchSRC_128):
	dcbt    r12,r4
	addi    r12,r12,128
	bdnz    L(prefetchSRC_128)


	/* Prefetching is done, or was not needed.

	   cr6 - are we on a cacheline boundary already?
	   r7  - number of quadwords to the next cacheline boundary
	*/

L(nocacheprefetch_128):
	mtctr   r7

	cmpldi  cr1,r5,128  /* Less than a cache line to copy? */

	/* How many bytes are left after we copy whatever full
	   cache lines we can get? */
	clrldi  r5,r5,64-7

	beq     cr6,L(cachelinealigned_128)


	/* Copy quadwords up to the next cacheline boundary */

L(aligntocacheline_128):
	ld      r9,0x08(r4)
	ld      r7,0x10(r4)
	addi    r4,r4,0x10
	std     r9,0x08(r6)
	stdu    r7,0x10(r6)
	bdnz    L(aligntocacheline_128)


L(cachelinealigned_128):        /* copy while cache lines  */

	blt-    cr1,L(lessthancacheline) /* size <128  */

L(outerloop_128):
	cmpdi   r11,0
	mtctr   r11
	beq-    L(endloop_128)

	li      r11,128*ZERO_AHEAD +8    /* DCBZ dist  */

	.align  4
	/* Copy whole cachelines, optimized by prefetching SRC cacheline  */
L(loop_128):                    /* Copy aligned body  */
	dcbt    r12,r4          /* PREFETCH SOURCE some cache lines ahead  */
	ld      r9, 0x08(r4)
	dcbz    r11,r6
	ld      r7, 0x10(r4)
	ld      r8, 0x18(r4)
	ld      r0, 0x20(r4)
	std     r9, 0x08(r6)
	std     r7, 0x10(r6)
	std     r8, 0x18(r6)
	std     r0, 0x20(r6)
	ld      r9, 0x28(r4)
	ld      r7, 0x30(r4)
	ld      r8, 0x38(r4)
	ld      r0, 0x40(r4)
	std     r9, 0x28(r6)
	std     r7, 0x30(r6)
	std     r8, 0x38(r6)
	std     r0, 0x40(r6)
	ld      r9, 0x48(r4)
	ld      r7, 0x50(r4)
	ld      r8, 0x58(r4)
	ld      r0, 0x60(r4)
	std     r9, 0x48(r6)
	std     r7, 0x50(r6)
	std     r8, 0x58(r6)
	std     r0, 0x60(r6)
	ld      r9, 0x68(r4)
	ld      r7, 0x70(r4)
	ld      r8, 0x78(r4)
	ld      r0, 0x80(r4)
	addi    r4, r4,0x80
	std     r9, 0x68(r6)
	std     r7, 0x70(r6)
	std     r8, 0x78(r6)
	stdu    r0, 0x80(r6)

	bdnz    L(loop_128)


L(endloop_128):
	cmpdi   r10,0
	beq-    L(endloop2_128)
	mtctr   r10

L(loop2_128):                       /* Copy aligned body  */
	ld      r9, 0x08(r4)
	ld      r7, 0x10(r4)
	ld      r8, 0x18(r4)
	ld      r0, 0x20(r4)
	std     r9, 0x08(r6)
	std     r7, 0x10(r6)
	std     r8, 0x18(r6)
	std     r0, 0x20(r6)
	ld      r9, 0x28(r4)
	ld      r7, 0x30(r4)
	ld      r8, 0x38(r4)
	ld      r0, 0x40(r4)
	std     r9, 0x28(r6)
	std     r7, 0x30(r6)
	std     r8, 0x38(r6)
	std     r0, 0x40(r6)
	ld      r9, 0x48(r4)
	ld      r7, 0x50(r4)
	ld      r8, 0x58(r4)
	ld      r0, 0x60(r4)
	std     r9, 0x48(r6)
	std     r7, 0x50(r6)
	std     r8, 0x58(r6)
	std     r0, 0x60(r6)
	ld      r9, 0x68(r4)
	ld      r7, 0x70(r4)
	ld      r8, 0x78(r4)
	ld      r0, 0x80(r4)
	addi    r4, r4,0x80
	std     r9, 0x68(r6)
	std     r7, 0x70(r6)
	std     r8, 0x78(r6)
	stdu    r0, 0x80(r6)

	bdnz    L(loop2_128)
L(endloop2_128):

	b       L(lessthancacheline)


END_GEN_TB (MEMCPY,TB_TOCLESS)
libc_hidden_builtin_def (memcpy)
