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

#define PREFETCH_AHEAD 4        /* no cache lines SRC prefetching ahead  */
#define ZERO_AHEAD 2            /* no cache lines DST zeroing ahead  */

	.machine  a2
EALIGN (memcpy, 5, 0)
	CALL_MCOUNT

	dcbt    0,r4            /* Prefetch ONE SRC cacheline  */
	cmplwi  cr1,r5,16       /* is size < 16 ?  */
	mr      r6,r3           /* Copy dest reg to r6; */
	blt+    cr1,L(shortcopy)


	/* Big copy (16 bytes or more)

	   Figure out how far to the nearest quadword boundary, or if we are
	   on one already.

	   r3 - return value (always)
	   r4 - current source addr
	   r5 - copy length
	   r6 - current dest addr
	*/

	neg     r8,r3           /* LS 4 bits = # bytes to 8-byte dest bdry  */
	clrlwi  r8,r8,32-4      /* align to 16byte boundary  */
	sub     r7,r4,r3        /* compute offset to src from dest */
	cmplwi  cr0,r8,0        /* Were we aligned on a 16 byte bdy? */
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
	lfdx    r0,r7,r6        /* copy 8 byte addr */
	stfd    r0,0(r6)
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


	.align  4
L(dst_aligned):


#ifdef PIC
	mflr    r0
/* Establishes GOT addressability so we can load the cache line size
   from rtld_global_ro.  This value was set from the aux vector during
   startup.  */
	SETUP_GOT_ACCESS(r9,got_label)
	addis	r9,r9,_GLOBAL_OFFSET_TABLE_-got_label@ha
	addi	r9,r9,_GLOBAL_OFFSET_TABLE_-got_label@l
	mtlr	r0
#endif
	__GLRO(r9, r9, _dl_cache_line_size,
	       RTLD_GLOBAL_RO_DL_CACHE_LINE_SIZE_OFFSET)

	cmplwi  cr5, r9, 0
	bne+    cr5,L(cachelineset)

/* Cache line size not set: generic byte copy without much optimization */
	andi.	r0,r5,1		/* If length is odd copy one byte.  */
	beq	L(cachelinenotset_align)
	lbz	r7,0(r4)	/* Read one byte from source.  */
	addi	r5,r5,-1	/* Update length.  */
	addi	r4,r4,1		/* Update source pointer address.  */
	stb	r7,0(r6)	/* Store one byte on dest.  */
	addi	r6,r6,1		/* Update dest pointer address.  */
L(cachelinenotset_align):
	cmpwi   cr7,r5,0	/* If length is 0 return.  */
	beqlr	cr7
	ori	r2,r2,0		/* Force a new dispatch group.  */
L(cachelinenotset_loop):
	addic.	r5,r5,-2	/* Update length.  */
	lbz	r7,0(r4)	/* Load 2 bytes from source.  */
	lbz	r8,1(r4)
	addi	r4,r4,2		/* Update source pointer address.  */
	stb	r7,0(r6)	/* Store 2 bytes on dest.  */
	stb	r8,1(r6)
	addi	r6,r6,2		/* Update dest pointer address.  */
	bne	L(cachelinenotset_loop)
	blr


L(cachelineset):

	addi   r10,r9,-1

	cmpw   cr5,r5,r10       /* Less than a cacheline to go? */

	neg     r7,r6           /* How far to next cacheline bdy? */

	addi    r6,r6,-8        /* prepare for stdu  */
	cmpwi   cr0,r9,128
	addi    r4,r4,-8        /* prepare for ldu  */


	ble+    cr5,L(lessthancacheline)

	beq-    cr0,L(big_lines) /* 128 byte line code */




	/* More than a cacheline left to go, and using 64 byte cachelines */

	clrlwi  r7,r7,32-6      /* How far to next cacheline bdy? */

	cmplwi  cr6,r7,0        /* Are we on a cacheline bdy already? */

	/* Reduce total len by what it takes to get to the next cache line */
	subf    r5,r7,r5
	srwi    r7,r7,4         /* How many qws to get to the line bdy? */

	/* How many full cache lines to copy after getting to a line bdy? */
	srwi    r10,r5,6

	cmplwi  r10,0           /* If no full cache lines to copy ... */
	li      r11,0           /* number cachelines to copy with prefetch  */
	beq     L(nocacheprefetch)


	/* We are here because we have at least one full cache line to copy,
	   and therefore some pre-touching to do. */

	cmplwi  r10,PREFETCH_AHEAD
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

	cmplwi  cr1,r5,64   /* Less than a cache line to copy? */

	/* How many bytes are left after we copy whatever full
	   cache lines we can get? */
	clrlwi  r5,r5,32-6

	beq     cr6,L(cachelinealigned)


	/* Copy quadwords up to the next cacheline boundary */

L(aligntocacheline):
	lfd     fp9,0x08(r4)
	lfdu    fp10,0x10(r4)
	stfd    fp9,0x08(r6)
	stfdu   fp10,0x10(r6)
	bdnz    L(aligntocacheline)


	.align 4
L(cachelinealigned):            /* copy while cache lines  */

	blt-    cr1,L(lessthancacheline) /* size <64  */

L(outerloop):
	cmpwi   r11,0
	mtctr   r11
	beq-    L(endloop)

	li      r11,64*ZERO_AHEAD +8    /* DCBZ dist  */

	.align  4
	/* Copy whole cachelines, optimized by prefetching SRC cacheline  */
L(loop):                        /* Copy aligned body  */
	dcbt    r12,r4          /* PREFETCH SOURCE some cache lines ahead  */
	lfd     fp9,  0x08(r4)
	dcbz    r11,r6
	lfd     fp10, 0x10(r4)
	lfd     fp11, 0x18(r4)
	lfd     fp12, 0x20(r4)
	stfd    fp9,  0x08(r6)
	stfd    fp10, 0x10(r6)
	stfd    fp11, 0x18(r6)
	stfd    fp12, 0x20(r6)
	lfd     fp9,  0x28(r4)
	lfd     fp10, 0x30(r4)
	lfd     fp11, 0x38(r4)
	lfdu    fp12, 0x40(r4)
	stfd    fp9,  0x28(r6)
	stfd    fp10, 0x30(r6)
	stfd    fp11, 0x38(r6)
	stfdu   fp12, 0x40(r6)

	bdnz    L(loop)


L(endloop):
	cmpwi   r10,0
	beq-    L(endloop2)
	mtctr   r10

L(loop2):                       /* Copy aligned body  */
	lfd     fp9,  0x08(r4)
	lfd     fp10, 0x10(r4)
	lfd     fp11, 0x18(r4)
	lfd     fp12, 0x20(r4)
	stfd    fp9,  0x08(r6)
	stfd    fp10, 0x10(r6)
	stfd    fp11, 0x18(r6)
	stfd    fp12, 0x20(r6)
	lfd     fp9,  0x28(r4)
	lfd     fp10, 0x30(r4)
	lfd     fp11, 0x38(r4)
	lfdu    fp12, 0x40(r4)
	stfd    fp9,  0x28(r6)
	stfd    fp10, 0x30(r6)
	stfd    fp11, 0x38(r6)
	stfdu   fp12, 0x40(r6)

	bdnz    L(loop2)
L(endloop2):


	.align  4
L(lessthancacheline):           /* Was there less than cache to do ?  */
	cmplwi  cr0,r5,16
	srwi    r7,r5,4         /* divide size by 16  */
	blt-    L(do_lt16)
	mtctr   r7

L(copy_remaining):
	lfd     fp9,  0x08(r4)
	lfdu    fp10, 0x10(r4)
	stfd    fp9,  0x08(r6)
	stfdu   fp10, 0x10(r6)
	bdnz    L(copy_remaining)

L(do_lt16):                     /* less than 16 ?  */
	cmplwi  cr0,r5,0        /* copy remaining bytes (0-15)  */
	beqlr+                  /* no rest to copy  */
	addi    r4,r4,8
	addi    r6,r6,8

L(shortcopy):                   /* SIMPLE COPY to handle size =< 15 bytes  */
	mtcrf   0x01,r5
	sub     r7,r4,r6
	bf-     cr7*4+0,8f
	lfdx    fp9,r7,r6       /* copy 8 byte  */
	stfd    fp9,0(r6)
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

	clrlwi  r7,r7,32-7      /* How far to next cacheline bdy? */

	cmplwi  cr6,r7,0        /* Are we on a cacheline bdy already? */

	/* Reduce total len by what it takes to get to the next cache line */
	subf    r5,r7,r5
	srwi    r7,r7,4         /* How many qw to get to the line bdy? */

	/* How many full cache lines to copy after getting to a line bdy? */
	srwi    r10,r5,7

	cmplwi  r10,0           /* If no full cache lines to copy ... */
	li      r11,0           /* number cachelines to copy with prefetch  */
	beq     L(nocacheprefetch_128)


	/* We are here because we have at least one full cache line to copy,
	   and therefore some pre-touching to do. */

	cmplwi  r10,PREFETCH_AHEAD
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

	cmplwi  cr1,r5,128  /* Less than a cache line to copy? */

	/* How many bytes are left after we copy whatever full
	   cache lines we can get? */
	clrlwi  r5,r5,32-7

	beq     cr6,L(cachelinealigned_128)


	/* Copy quadwords up to the next cacheline boundary */

L(aligntocacheline_128):
	lfd     fp9,0x08(r4)
	lfdu    fp10,0x10(r4)
	stfd    fp9,0x08(r6)
	stfdu   fp10,0x10(r6)
	bdnz    L(aligntocacheline_128)


L(cachelinealigned_128):        /* copy while cache lines  */

	blt-    cr1,L(lessthancacheline) /* size <128  */

L(outerloop_128):
	cmpwi   r11,0
	mtctr   r11
	beq-    L(endloop_128)

	li      r11,128*ZERO_AHEAD +8    /* DCBZ dist  */

	.align  4
	/* Copy whole cachelines, optimized by prefetching SRC cacheline  */
L(loop_128):                    /* Copy aligned body  */
	dcbt    r12,r4          /* PREFETCH SOURCE some cache lines ahead  */
	lfd     fp9,  0x08(r4)
	dcbz    r11,r6
	lfd     fp10, 0x10(r4)
	lfd     fp11, 0x18(r4)
	lfd     fp12, 0x20(r4)
	stfd    fp9,  0x08(r6)
	stfd    fp10, 0x10(r6)
	stfd    fp11, 0x18(r6)
	stfd    fp12, 0x20(r6)
	lfd     fp9,  0x28(r4)
	lfd     fp10, 0x30(r4)
	lfd     fp11, 0x38(r4)
	lfd     fp12, 0x40(r4)
	stfd    fp9,  0x28(r6)
	stfd    fp10, 0x30(r6)
	stfd    fp11, 0x38(r6)
	stfd    fp12, 0x40(r6)
	lfd     fp9,  0x48(r4)
	lfd     fp10, 0x50(r4)
	lfd     fp11, 0x58(r4)
	lfd     fp12, 0x60(r4)
	stfd    fp9,  0x48(r6)
	stfd    fp10, 0x50(r6)
	stfd    fp11, 0x58(r6)
	stfd    fp12, 0x60(r6)
	lfd     fp9,  0x68(r4)
	lfd     fp10, 0x70(r4)
	lfd     fp11, 0x78(r4)
	lfdu    fp12, 0x80(r4)
	stfd    fp9,  0x68(r6)
	stfd    fp10, 0x70(r6)
	stfd    fp11, 0x78(r6)
	stfdu   fp12, 0x80(r6)

	bdnz    L(loop_128)


L(endloop_128):
	cmpwi   r10,0
	beq-    L(endloop2_128)
	mtctr   r10

L(loop2_128):                   /* Copy aligned body  */
	lfd     fp9,  0x08(r4)
	lfd     fp10, 0x10(r4)
	lfd     fp11, 0x18(r4)
	lfd     fp12, 0x20(r4)
	stfd    fp9,  0x08(r6)
	stfd    fp10, 0x10(r6)
	stfd    fp11, 0x18(r6)
	stfd    fp12, 0x20(r6)
	lfd     fp9,  0x28(r4)
	lfd     fp10, 0x30(r4)
	lfd     fp11, 0x38(r4)
	lfd     fp12, 0x40(r4)
	stfd    fp9,  0x28(r6)
	stfd    fp10, 0x30(r6)
	stfd    fp11, 0x38(r6)
	stfd    fp12, 0x40(r6)
	lfd     fp9,  0x48(r4)
	lfd     fp10, 0x50(r4)
	lfd     fp11, 0x58(r4)
	lfd     fp12, 0x60(r4)
	stfd    fp9,  0x48(r6)
	stfd    fp10, 0x50(r6)
	stfd    fp11, 0x58(r6)
	stfd    fp12, 0x60(r6)
	lfd     fp9,  0x68(r4)
	lfd     fp10, 0x70(r4)
	lfd     fp11, 0x78(r4)
	lfdu    fp12, 0x80(r4)
	stfd    fp9,  0x68(r6)
	stfd    fp10, 0x70(r6)
	stfd    fp11, 0x78(r6)
	stfdu   fp12, 0x80(r6)
	bdnz    L(loop2_128)
L(endloop2_128):

	b       L(lessthancacheline)


END (memcpy)
libc_hidden_builtin_def (memcpy)
