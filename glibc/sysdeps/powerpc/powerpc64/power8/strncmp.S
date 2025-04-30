/* Optimized strncmp implementation for PowerPC64/POWER8.
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

#ifndef STRNCMP
# define STRNCMP strncmp
#endif

/* Implements the function

   int [r3] strncmp (const char *s1 [r3], const char *s2 [r4], size_t [r5] n)

   The implementation uses unaligned doubleword access to avoid specialized
   code paths depending of data alignment.  Although recent powerpc64 uses
   64K as default, the page cross handling assumes minimum page size of
   4k.  */

	.machine  power8
ENTRY_TOCLESS (STRNCMP, 4)
	/* Check if size is 0.  */
	mr.	r10,r5
	beq	cr0,L(ret0)

	/* Check if [s1]+16 or [s2]+16 will cross a 4K page boundary using
	   the code:

	    (((size_t) s1) % PAGE_SIZE > (PAGE_SIZE - ITER_SIZE))

	   with PAGE_SIZE being 4096 and ITER_SIZE begin 16.  */
	rldicl	r8,r3,0,52
	cmpldi	cr7,r8,4096-16
	bgt	cr7,L(pagecross)
	rldicl	r9,r4,0,52
	cmpldi	cr7,r9,4096-16
	bgt	cr7,L(pagecross)

	/* For short string up to 16 bytes, load both s1 and s2 using
	   unaligned dwords and compare.  */
	ld	r7,0(r3)
	ld	r9,0(r4)
	li	r8,0
	cmpb	r8,r7,r8
	cmpb	r6,r7,r9
	orc.	r8,r8,r6
	bne	cr0,L(different1)

	/* If the string compared are equal, but size is less or equal
	   to 8, return 0.  */
	cmpldi	cr7,r10,8
	li	r9,0
	ble	cr7,L(ret1)
	addi	r5,r10,-8

	ld	r7,8(r3)
	ld	r9,8(r4)
	cmpb	r8,r7,r8
	cmpb	r6,r7,r9
	orc.	r8,r8,r6
	bne	cr0,L(different0)

	cmpldi	cr7,r5,8
	mr	r9,r8
	ble	cr7,L(ret1)

	/* Update pointers and size.  */
	addi	r10,r10,-16
	addi	r3,r3,16
	addi	r4,r4,16

	/* Now it has checked for first 16 bytes, align source1 to doubleword
	   and adjust source2 address.  */
L(align_8b):
	rldicl	r5,r3,0,61
	rldicr	r3,r3,0,60
	subf	r4,r5,r4
	add	r10,r10,r5

	/* At this point, source1 alignment is 0 and source2 alignment is
	   between 0 and 7.  Check is source2 alignment is 0, meaning both
	   sources have the same alignment.  */
	andi.	r8,r4,0x7
	beq	cr0,L(loop_eq_align_0)

	li	r5,0
	b	L(loop_ne_align_1)

	/* If source2 is unaligned to doubleword, the code needs to check
	   on each interation if the unaligned doubleword access will cross
	   a 4k page boundary.  */
	.align 4
L(loop_ne_align_0):
	ld	r7,0(r3)
	ld	r9,0(r4)
	cmpb	r8,r7,r5
	cmpb	r6,r7,r9
	orc.	r8,r8,r6
	bne	cr0,L(different1)

	cmpldi	cr7,r10,8
	ble	cr7,L(ret0)
	addi	r10,r10,-8
	addi	r3,r3,8
	addi	r4,r4,8
L(loop_ne_align_1):
	rldicl	r9,r4,0,52
	cmpldi	r7,r9,4088
	ble	cr7,L(loop_ne_align_0)
	cmpdi	cr7,r10,0
	beq	cr7,L(ret0)

	lbz	r9,0(r3)
	lbz	r8,0(r4)
	cmplw	cr7,r9,r8
	bne	cr7,L(byte_ne_4)
	cmpdi	cr7,r9,0
	beq	cr7,L(size_reached_0)

	li	r9,r7
	addi	r8,r3,1
	mtctr	r9
	addi	r4,r4,1
	addi	r10,r10,-1
	addi	r3,r3,8

	/* The unaligned read of source2 will cross a 4K page boundary,
	   and the different byte or NULL maybe be in the remaining page
	   bytes.  Since it can not use the unaligned load the algorithm
	   reads and compares 8 bytes to keep source1 doubleword aligned.  */
	.align 4
L(loop_ne_align_byte):
	cmpdi	cr7,r10,0
	addi	r10,r10,-1
	beq	cr7,L(ret0)
	lbz	r9,0(r8)
	lbz	r7,0(r4)
	addi	r8,r8,1
	addi	r4,r4,1
	cmplw	cr7,r9,r7
	cmpdi	cr5,r9,0
	bne	cr7,L(size_reached_2)
	beq	cr5,L(size_reached_0)
	bdnz	L(loop_ne_align_byte)

	cmpdi	cr7,r10,0
	bne+	cr7,L(loop_ne_align_0)

	.align 4
L(ret0):
	li	r9,0
L(ret1):
	mr	r3,r9
	blr

	/* The code now check if r8 and r10 are different by issuing a
	   cmpb and shift the result based on its output:

	#ifdef __LITTLE_ENDIAN__
	  leadzero = (__builtin_ffsl (z1) - 1);
	  leadzero = leadzero > (n-1)*8 ? (n-1)*8 : leadzero;
	  r1 = (r1 >> leadzero) & 0xFFUL;
	  r2 = (r2 >> leadzero) & 0xFFUL;
	#else
	  leadzero = __builtin_clzl (z1);
	  leadzero = leadzero > (n-1)*8 ? (n-1)*8 : leadzero;
	  r1 = (r1 >> (56 - leadzero)) & 0xFFUL;
	  r2 = (r2 >> (56 - leadzero)) & 0xFFUL;
	#endif
	  return r1 - r2;  */

	.align 4
L(different0):
	mr	r10,r5
#ifdef __LITTLE_ENDIAN__
L(different1):
        neg	r11,r8
        sldi	r10,r10,3
        and	r8,r11,r8
        addi	r10,r10,-8
        cntlzd	r8,r8
        subfic	r8,r8,63
        extsw 	r8,r8
        cmpld	cr7,r8,r10
        ble	cr7,L(different2)
        mr	r8,r10
L(different2):
        extsw	r8,r8
#else
L(different1):
	addi	r10,r10,-1
	cntlzd	r8,r8
	sldi	r10,r10,3
	cmpld	cr7,r8,r10
	blt	cr7,L(different2)
	mr	r8,r10
L(different2):
	subfic	r8,r8,56
#endif
	srd	r7,r7,r8
	srd	r9,r9,r8
	rldicl	r3,r7,0,56
	rldicl	r9,r9,0,56
	subf	r9,r9,3
	extsw	r9,r9
	mr	r3,r9
	blr

	/* If unaligned 16 bytes reads across a 4K page boundary, it uses
	   a simple byte a byte comparison until the page alignment for s1
	   is reached.  */
	.align 4
L(pagecross):
	lbz	r7,0(r3)
	lbz	r9,0(r4)
	subfic	r8,r8,4095
	cmplw	cr7,r9,r7
	bne	cr7,L(byte_ne_3)
	cmpdi	cr7,r9,0
	beq	cr7,L(byte_ne_0)
	addi	r10,r10,-1
	subf	r7,r8,r10
	subf	r9,r7,r10
	addi	r9,r9,1
	mtctr	r9
	b	L(pagecross_loop1)

	.align 4
L(pagecross_loop0):
	beq	cr7,L(ret0)
	lbz	r9,0(r3)
	lbz	r8,0(r4)
	addi	r10,r10,-1
	cmplw	cr7,r9,r8
	cmpdi	cr5,r9,0
	bne	r7,L(byte_ne_2)
	beq	r5,L(byte_ne_0)
L(pagecross_loop1):
	cmpdi	cr7,r10,0
	addi	r3,r3,1
	addi	r4,r4,1
	bdnz	L(pagecross_loop0)
	cmpdi	cr7,r7,0
	li	r9,0
	bne+	cr7,L(align_8b)
	b	L(ret1)

	/* If both source1 and source2 are doubleword aligned, there is no
	   need for page boundary cross checks.  */
	.align 4
L(loop_eq_align_0):
	ld	r7,0(r3)
	ld	r9,0(r4)
	cmpb	r8,r7,r8
	cmpb	r6,r7,r9
	orc.	r8,r8,r6
	bne	cr0,L(different1)

	cmpldi	cr7,r10,8
	ble	cr7,L(ret0)
	addi	r9,r10,-9

	li	r5,0
	srdi	r9,r9,3
	addi	r9,r9,1
	mtctr	r9
	b	L(loop_eq_align_2)

	.align 4
L(loop_eq_align_1):
	bdz	L(ret0)
L(loop_eq_align_2):
	ldu	r7,8(r3)
	addi	r10,r10,-8
	ldu	r9,8(r4)
	cmpb	r8,r7,r5
	cmpb	r6,r7,r9
	orc.	r8,r8,r6
	beq	cr0,L(loop_eq_align_1)
	b	L(different1)

	.align 4
L(byte_ne_0):
	li	r7,0
L(byte_ne_1):
	subf	r9,r9,r7
	extsw	r9,r9
	b	L(ret1)

	.align 4
L(byte_ne_2):
	extsw	r7,r9
	mr	r9,r8
	b	L(byte_ne_1)
L(size_reached_0):
	li	r10,0
L(size_reached_1):
	subf	r9,r9,r10
	extsw	r9,r9
	b	L(ret1)
L(size_reached_2):
	extsw	r10,r9
	mr	r9,r7
	b	L(size_reached_1)
L(byte_ne_3):
	extsw	r7,r7
	b	L(byte_ne_1)
L(byte_ne_4):
	extsw	r10,r9
	mr	r9,r8
	b	L(size_reached_1)
END(STRNCMP)
libc_hidden_builtin_def(strncmp)
