/* Vector Optimized 32/64 bit S/390 version of wmemcmp.
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

#include <ifunc-wmemcmp.h>
#if HAVE_WMEMCMP_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* int wmemcmp (const wchar_t *s1, const wchar_t *s2, size_t n)
   Compare at most n characters of two wchar_t-arrays.

   Register usage:
   -r0=tmp
   -r1=number of blocks
   -r2=s1
   -r3=s2
   -r4=n
   -r5=current_len
   -v16=part of s1
   -v17=part of s2
   -v18=index of unequal
*/
ENTRY(WMEMCMP_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

# if !defined __s390x__
	llgfr	%r4,%r4
# endif /* !defined __s390x__ */
	clgije	%r4,0,.Lend_equal /* Nothing to do if n == 0.  */

	/* Check range of maxlen and convert to byte-count.  */
# ifdef __s390x__
	tmhh	%r4,49152	/* Test bit 0 or 1 of maxlen.  */
	lghi	%r1,-4		/* Max byte-count is 18446744073709551612.  */
# else
	tmlh	%r4,49152	/* Test bit 0 or 1 of maxlen.  */
	llilf	%r1,4294967292	/* Max byte-count is 4294967292.  */
# endif /* !__s390x__ */
	sllg	%r4,%r4,2	/* Convert character-count to byte-count.  */
	locgrne	%r4,%r1		/* Use max byte-count, if bit 0/1 was one.  */

	lghi	%r5,0		/* current_len = 0.  */

	clgijh	%r4,16,.Lgt16

.Lremaining:
	aghi	%r4,-1		/* vstl needs highest index.  */
	vll	%v16,%r4,0(%r2)
	vll	%v17,%r4,0(%r3)
	vfenef	%v18,%v16,%v17	/* Compare not equal.  */
	vlgvb	%r1,%v18,7	/* Load unequal index or 16 if not found.  */
	clrj	%r1,%r4,12,.Lfound2 /* r1 <= r4 -> unequal within loaded
					bytes.  */

.Lend_equal:
	lghi	%r2,0
	br	%r14

.Lfound:
	/* vfenezf found an unequal element or zero.
	   This instruction compares unsigned words, but wchar_t is signed.
	   Thus we have to compare the found element again.  */
	vlgvb	%r1,%v18,7	/* Extract not equal byte-index.  */
.Lfound2:
	srl	%r1,2		/* And convert it to character-index.  */
	vlgvf	%r0,%v16,0(%r1)	/* Load character-values.  */
	vlgvf	%r1,%v17,0(%r1)
	cr	%r0,%r1
	je	.Lend_equal
	lghi	%r2,1
	lghi	%r1,-1
	locgrl	%r2,%r1
	br	%r14

.Lgt16:
	clgijh	%r4,64,.Lpreloop64

.Lpreloop16:
	srlg	%r1,%r4,4	/* Split into 16byte blocks */
.Lloop16:
	vl	%v16,0(%r5,%r2)
	vl	%v17,0(%r5,%r3)
	aghi	%r5,16
	vfenefs	%v18,%v16,%v17	/* Compare not equal.  */
	jno	.Lfound
	brctg	%r1,.Lloop16	/* Loop until all blocks are processed.  */

	llgfr	%r4,%r4
	nilf	%r4,15		/* Get remaining bytes */
	locgre	%r2,%r4
	ber	%r14
	la	%r2,0(%r5,%r2)
	la	%r3,0(%r5,%r3)
	j	.Lremaining

.Lpreloop64:
	srlg	%r1,%r4,6	/* Split into 64byte blocks */
.Lloop64:
	vl	%v16,0(%r5,%r2)
	vl	%v17,0(%r5,%r3)
	vfenefs	%v18,%v16,%v17	/* Compare not equal.  */
	jno	.Lfound

	vl	%v16,16(%r5,%r2)
	vl	%v17,16(%r5,%r3)
	vfenefs	%v18,%v16,%v17
	jno	.Lfound

	vl	%v16,32(%r5,%r2)
	vl	%v17,32(%r5,%r3)
	vfenefs	%v18,%v16,%v17
	jno	.Lfound

	vl	%v16,48(%r5,%r2)
	vl	%v17,48(%r5,%r3)
	aghi	%r5,64
	vfenefs	%v18,%v16,%v17
	jno	.Lfound

	brctg	%r1,.Lloop64	/* Loop until all blocks are processed.  */

	llgfr	%r4,%r4
	nilf	%r4,63		/* Get remaining bytes */
	locgre	%r2,%r4
	ber	%r14
	clgijh	%r4,16,.Lpreloop16
	la	%r2,0(%r5,%r2)
	la	%r3,0(%r5,%r3)
	j	.Lremaining
END(WMEMCMP_Z13)

# if ! HAVE_WMEMCMP_IFUNC
strong_alias (WMEMCMP_Z13, __wmemcmp)
weak_alias (__wmemcmp, wmemcmp)
# endif
#endif
