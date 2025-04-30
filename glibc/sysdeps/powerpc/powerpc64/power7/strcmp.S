/* Optimized strcmp implementation for Power7 using 'cmpb' instruction
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

/* The optimization is achieved here through cmpb instruction.
   8byte aligned strings are processed with double word comparision
   and unaligned strings are handled effectively with loop unrolling
   technique  */

#include <sysdep.h>

#ifndef STRCMP
# define STRCMP strcmp
#endif

/* int [r3] strcmp (const char *s1 [r3], const char *s2 [r4])  */

	.machine	power7
ENTRY_TOCLESS (STRCMP, 4)
	CALL_MCOUNT 2

	or r9, r3, r4
	rldicl. r10, r9, 0, 61	/* are s1 and s2 8 byte aligned..?  */
	bne cr0, L(process_unaligned_bytes)
	li	r5, 0

	.align 4
/* process input parameters on double word aligned boundary  */
L(unrollDword):
	ld	r8,0(r3)
	ld	r10,0(r4)
	cmpb	r7,r8,r5
	cmpdi	cr7,r7,0
	mr	r9,r7
	bne 	cr7,L(null_found)
	cmpld	cr7,r8,r10
	bne	cr7,L(different)

	ld	r8,8(r3)
	ld	r10,8(r4)
	cmpb	r7,r8,r5
	cmpdi	cr7,r7,0
	mr	r9,r7
	bne 	cr7,L(null_found)
	cmpld	cr7,r8,r10
	bne	cr7,L(different)

	ld	r8,16(r3)
	ld	r10,16(r4)
	cmpb	r7,r8,r5
	cmpdi	cr7,r7,0
	mr	r9,r7
	bne 	cr7,L(null_found)
	cmpld	cr7,r8,r10
	bne	cr7,L(different)

	ld	r8,24(r3)
	ld	r10,24(r4)
	cmpb	r7,r8,r5
	cmpdi	cr7,r7,0
	mr	r9,r7
	bne 	cr7,L(null_found)
	cmpld	cr7,r8,r10
	bne	cr7,L(different)

	addi r3, r3, 32
	addi r4, r4, 32
	beq cr7, L(unrollDword)

	.align 4
L(null_found):
#ifdef __LITTLE_ENDIAN__
	neg	r7,r9
	and	r9,r9,r7
	li	r7,-1
	cntlzd	r9,r9
	subfic	r9,r9,71
	sld	r9,r7,r9
#else
	cntlzd	r9,r9
	li	r7,-1
	addi	r9,r9,8
	srd	r9,r7,r9
#endif
	or	r8,r8,r9
	or	r10,r10,r9

L(different):
	cmpb	r9,r8,r10
#ifdef __LITTLE_ENDIAN__
	addi	r7,r9,1
	andc	r9,r7,r9
	cntlzd	r9,r9
	subfic	r9,r9,63
#else
	not	r9,r9
	cntlzd	r9,r9
	subfic	r9,r9,56
#endif
	srd	r3,r8,r9
	srd	r10,r10,r9
	rldicl	r10,r10,0,56
	rldicl	r3,r3,0,56
	subf	r3,r10,r3
	blr

	.align 4
L(process_unaligned_bytes):
	lbz r9, 0(r3)		/* load byte from s1  */
	lbz r10, 0(r4)		/* load byte from s2  */
	cmpdi cr7, r9, 0	/* compare *s1 with NULL  */
	beq cr7, L(diffOfNULL)	/* if *s1 is NULL , return *s1 - *s2  */
	cmplw cr7, r9, r10	/* compare *s1 and *s2  */
	bne cr7, L(ComputeDiff)	/* branch to compute difference and return  */

	lbz r9, 1(r3)		/* load next byte from s1  */
	lbz r10, 1(r4)		/* load next byte from s2  */
	cmpdi cr7, r9, 0	/* compare *s1 with NULL  */
	beq cr7, L(diffOfNULL)	/* if *s1 is NULL , return *s1 - *s2  */
	cmplw cr7, r9, r10	/* compare *s1 and *s2  */
	bne cr7, L(ComputeDiff)	/* branch to compute difference and return  */

	lbz r9, 2(r3)		/* unroll 3rd byte here  */
	lbz r10, 2(r4)
	cmpdi cr7, r9, 0
	beq cr7, L(diffOfNULL)
	cmplw cr7, r9, r10
	bne 7, L(ComputeDiff)

	lbz r9, 3(r3)		/* unroll 4th byte now  */
	lbz r10, 3(r4)
	addi r3, r3, 4		/* increment s1 by unroll factor  */
	cmpdi cr7, r9, 0
	cmplw cr6, 9, r10
	beq cr7, L(diffOfNULL)
	addi r4, r4, 4		/* increment s2 by unroll factor  */
	beq cr6, L(process_unaligned_bytes)	/* unroll byte processing  */

	.align 4
L(ComputeDiff):
	extsw r9, r9
	subf r10, r10, r9	/* compute s1 - s2  */
	extsw r3, r10
	blr			/* return  */

	.align 4
L(diffOfNULL):
	li r9, 0
	subf r10, r10, r9	/* compute s1 - s2  */
	extsw r3, r10		/* sign extend result  */
	blr			/* return  */

END (STRCMP)
libc_hidden_builtin_def (strcmp)
