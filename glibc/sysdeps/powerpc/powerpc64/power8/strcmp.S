/* Optimized strcmp implementation for PowerPC64/POWER8.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#ifndef STRCMP
# define STRCMP strcmp
#endif

/* Implements the function

   size_t [r3] strcmp (const char *s1 [r3], const char *s2 [r4])

   The implementation uses unaligned doubleword access to avoid specialized
   code paths depending of data alignment.  Although recent powerpc64 uses
   64K as default, the page cross handling assumes minimum page size of
   4k.  */

	.machine power8
ENTRY_TOCLESS (STRCMP, 4)
	li	r0,0

	/* Check if [s1]+16 or [s2]+16 will cross a 4K page boundary using
	   the code:

	    (((size_t) s1) % PAGE_SIZE > (PAGE_SIZE - ITER_SIZE))

	   with PAGE_SIZE being 4096 and ITER_SIZE begin 16.  */

	rldicl	r7,r3,0,52
	rldicl	r9,r4,0,52
	cmpldi	cr7,r7,4096-16
	bgt	cr7,L(pagecross_check)
	cmpldi	cr5,r9,4096-16
	bgt	cr5,L(pagecross_check)

	/* For short string up to 16 bytes, load both s1 and s2 using
	   unaligned dwords and compare.  */
	ld	r8,0(r3)
	ld	r10,0(r4)
	cmpb	r12,r8,r0
	cmpb	r11,r8,r10
	orc.	r9,r12,r11
	bne	cr0,L(different_nocmpb)

	ld	r8,8(r3)
	ld	r10,8(r4)
	cmpb	r12,r8,r0
	cmpb	r11,r8,r10
	orc.	r9,r12,r11
	bne	cr0,L(different_nocmpb)

	addi	r7,r3,16
	addi	r4,r4,16

L(align_8b):
	/* Now it has checked for first 16 bytes, align source1 to doubleword
	   and adjust source2 address.  */
	rldicl	r9,r7,0,61	/* source1 alignment to doubleword  */
	subf	r4,r9,r4	/* Adjust source2 address based on source1
				   alignment.  */
	rldicr	r7,r7,0,60	/* Align source1 to doubleword.  */

	/* At this point, source1 alignment is 0 and source2 alignment is
	   between 0 and 7.  Check is source2 alignment is 0, meaning both
	   sources have the same alignment.  */
	andi.	r9,r4,0x7
	bne	cr0,L(loop_diff_align)

	/* If both source1 and source2 are doubleword aligned, there is no
	   need for page boundary cross checks.  */

	ld	r8,0(r7)
	ld	r10,0(r4)
	cmpb	r12,r8,r0
	cmpb	r11,r8,r10
	orc.	r9,r12,r11
	bne	cr0,L(different_nocmpb)

	.align 4
L(loop_equal_align):
	ld	r8,8(r7)
	ld	r10,8(r4)
	cmpb	r12,r8,r0
	cmpb	r11,r8,r10
	orc.	r9,r12,r11
	bne	cr0,L(different_nocmpb)

	ld	r8,16(r7)
	ld	r10,16(r4)
	cmpb	r12,r8,r0
	cmpb	r11,r8,r10
	orc.	r9,r12,r11
	bne	cr0,L(different_nocmpb)

	ldu	r8,24(r7)
	ldu	r10,24(r4)
	cmpb	r12,r8,r0
	cmpb	r11,r8,r10
	orc.	r9,r12,r11
	bne	cr0,L(different_nocmpb)

	b	L(loop_equal_align)

	/* A zero byte was found in r8 (s1 dword), r9 contains the cmpb
	   result and r10 the dword from s2.  To code isolate the byte
	   up to end (including the '\0'), masking with 0xFF the remaining
	   ones:

           #if __LITTLE_ENDIAN__
	     (__builtin_ffsl (x) - 1) = counting trailing zero bits
	     r9 = (__builtin_ffsl (r9) - 1) + 8;
	     r9 = -1UL << r9
	   #else
	     r9  = __builtin_clzl (r9) + 8;
	     r9  = -1UL >> r9
	   #endif
	     r8  = r8  | r9
	     r10 = r10 | r9  */

#ifdef __LITTLE_ENDIAN__
	nor 	r9,r9,r9
L(different_nocmpb):
	neg	r3,r9
	and	r9,r9,r3
	cntlzd	r9,r9
	subfic	r9,r9,63
#else
	not	r9,r9
L(different_nocmpb):
	cntlzd	r9,r9
	subfic	r9,r9,56
#endif
	srd	r3,r8,r9
	srd	r10,r10,r9
	rldicl	r10,r10,0,56
	rldicl	r3,r3,0,56
	subf	r3,r10,r3
	extsw	r3,r3
	blr

	.align	4
L(pagecross_check):
	subfic	r9,r9,4096
	subfic	r7,r7,4096
	cmpld	cr7,r7,r9
	bge	cr7,L(pagecross)
	mr	r7,r9

	/* If unaligned 16 bytes reads across a 4K page boundary, it uses
	   a simple byte a byte comparison until the page alignment for s1
	   is reached.  */
L(pagecross):
	add	r7,r3,r7
	subf	r9,r3,r7
	mtctr	r9

	.align	4
L(pagecross_loop):
	/* Loads a byte from s1 and s2, compare if *s1 is equal to *s2
	   and if *s1 is '\0'.  */
	lbz	r9,0(r3)
	lbz	r10,0(r4)
	addi	r3,r3,1
	addi	r4,r4,1
	cmplw	cr7,r9,r10
	cmpdi	cr5,r9,r0
	bne	cr7,L(pagecross_ne)
	beq	cr5,L(pagecross_nullfound)
	bdnz	L(pagecross_loop)
	b	L(align_8b)

	.align	4
	/* The unaligned read of source2 will cross a 4K page boundary,
	   and the different byte or NULL maybe be in the remaining page
	   bytes. Since it can not use the unaligned load, the algorithm
	   reads and compares 8 bytes to keep source1 doubleword aligned.  */
L(check_source2_byte):
	li	r9,8
	mtctr	r9

	.align	4
L(check_source2_byte_loop):
	lbz	r9,0(r7)
	lbz	r10,0(r4)
	addi	r7,r7,1
	addi	r4,r4,1
	cmplw	cr7,r9,10
	cmpdi	r5,r9,0
	bne	cr7,L(pagecross_ne)
	beq	cr5,L(pagecross_nullfound)
	bdnz	L(check_source2_byte_loop)

	/* If source2 is unaligned to doubleword, the code needs to check
	   on each interation if the unaligned doubleword access will cross
	   a 4k page boundary.  */
	.align	5
L(loop_unaligned):
	ld	r8,0(r7)
	ld	r10,0(r4)
	cmpb	r12,r8,r0
	cmpb	r11,r8,r10
	orc.	r9,r12,r11
	bne	cr0,L(different_nocmpb)
	addi	r7,r7,8
	addi	r4,r4,8

L(loop_diff_align):
	/* Check if [src2]+8 cross a 4k page boundary:

	     srcin2 % PAGE_SIZE > (PAGE_SIZE - 8)

	     with PAGE_SIZE being 4096.  */
	rldicl	r9,r4,0,52
	cmpldi	cr7,r9,4088
	ble	cr7,L(loop_unaligned)
	b	L(check_source2_byte)

	.align	4
L(pagecross_ne):
	extsw	r3,r9
	mr	r9,r10
L(pagecross_retdiff):
	subf	r9,r9,r3
	extsw	r3,r9
	blr

	.align	4
L(pagecross_nullfound):
	li	r3,0
	b	L(pagecross_retdiff)
END (STRCMP)
libc_hidden_builtin_def (strcmp)
