/* Vector optimized 32/64 bit S/390 version of wcsncmp.
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

#include <ifunc-wcsncmp.h>
#if HAVE_WCSNCMP_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* int wcsncmp (const wchar_t *s1, const wchar_t *s2, size_t n)
   Compare at most n characters of two strings.

   Register usage:
   -r0=tmp
   -r1=tmp
   -r2=s1
   -r3=s2
   -r4=n
   -r5=current_len
   -v16=part of s1
   -v17=part of s2
   -v18=index of unequal
*/
ENTRY(WCSNCMP_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

# if !defined __s390x__
	llgfr	%r4,%r4
# endif /* !defined __s390x__ */

	clgije	%r4,0,.Lend_equal /* Nothing to do if n == 0.  */

	/* Check range of n and convert to byte-count.  */
# ifdef __s390x__
	tmhh	%r4,49152	/* Test bit 0 or 1 of maxlen.  */
	lghi	%r1,-4		/* Max byte-count is 18446744073709551612.  */
# else
	tmlh	%r4,49152	/* Test bit 0 or 1 of maxlen.  */
	llilf	%r1,4294967292	/* Max byte-count is 4294967292.  */
# endif /* !__s390x__ */
	sllg	%r4,%r4,2	/* Convert character-count to byte-count.  */
	locgrne	%r4,%r1		/* Use max byte-count, if bit 0/1 was one.  */

	/* Check first character without vector load.  */
	lghi	%r5,4		/* current_len = 4 bytes.  */
	/* Check s1/2[0].  */
	lt	%r0,0(%r2)
	l	%r1,0(%r3)
	je	.Lend_cmp_one_char
	crjne	%r0,%r1,.Lend_cmp_one_char

.Lloop:
	vlbb	%v17,0(%r5,%r3),6 /* Load s2 to block boundary.  */
	vlbb	%v16,0(%r5,%r2),6 /* Load s1 to block boundary.  */
	lcbb	%r0,0(%r5,%r2),6 /* Get loaded byte count of s1.  */
	jo	.Llt16_1	/* Jump away if vector not fully loaded.  */
	lcbb	%r1,0(%r5,%r3),6 /* Get loaded byte count of s2.  */
	jo	.Llt16_2	/* Jump away if vector not fully loaded.  */
	aghi	%r5,16		/* Both vectors are fully loaded.  */
	vfenezfs %v18,%v16,%v17	/* Compare not equal with zero search.  */
	clgrjhe	%r5,%r4,.Llastcmp /* If current_len >= n ->last compare.  */
	jno	.Lfound

	vlbb	%v17,0(%r5,%r3),6
	vlbb	%v16,0(%r5,%r2),6
	lcbb	%r0,0(%r5,%r2),6
	jo	.Llt16_1
	lcbb	%r1,0(%r5,%r3),6
	jo	.Llt16_2
	aghi	%r5,16
	vfenezfs %v18,%v16,%v17
	clgrjhe	%r5,%r4,.Llastcmp
	jno	.Lfound

	vlbb	%v17,0(%r5,%r3),6
	vlbb	%v16,0(%r5,%r2),6
	lcbb	%r0,0(%r5,%r2),6
	jo	.Llt16_1
	lcbb	%r1,0(%r5,%r3),6
	jo	.Llt16_2
	aghi	%r5,16
	vfenezfs %v18,%v16,%v17
	clgrjhe	%r5,%r4,.Llastcmp
	jno	.Lfound

	vlbb	%v17,0(%r5,%r3),6
	vlbb	%v16,0(%r5,%r2),6
	lcbb	%r0,0(%r5,%r2),6
	jo	.Llt16_1
	lcbb	%r1,0(%r5,%r3),6
	jo	.Llt16_2
	aghi	%r5,16
	vfenezfs %v18,%v16,%v17
	clgrjhe	%r5,%r4,.Llastcmp
	jno	.Lfound

	j	.Lloop

.Llt16_1:
	lcbb	%r1,0(%r5,%r3),6 /* Get loaded byte count of s2.  */
.Llt16_2:
	clr	%r0,%r1		/* Compare logical.  */
	locrh	%r0,%r1		/* Compute minimum of bytes loaded.  */
	nill	%r0,65532	/* Align bytes loaded to full characters.  */
	jz	.Lcmp_one_char	/* Jump away if no full char is available.  */
.Llt_cmp:
	algfr	%r5,%r0		/* Add smallest loaded bytes to current_len.  */
	vfenezfs %v18,%v16,%v17	/* Compare not equal with zero search.  */
	clgrj	%r5,%r4,10,.Llastcmp /* If current_len >= n -> last compare  */
	vlgvb	%r1,%v18,7	/* Get not equal index or 16 if all equal.  */
	clrjl	%r1,%r0,.Lfound /* Jump away if miscompare is within
				   loaded bytes; (index < loaded-bytes) */
	j	.Lloop

.Lcmp_one_char:
	/* At least one of both strings is not 4-byte aligned
	   and there is no full character before next block-boundary.
	   Compare one character to get over the boundary and
	   proceed with normal loop!  */
	vlef	%v16,0(%r5,%r2),0 /* Load one character.  */
	lghi	%r0,4		/* Loaded byte count is 4.  */
	vlef	%v17,0(%r5,%r3),0
	j	.Llt_cmp	/* Proceed with comparision.  */

.Llastcmp:
	/* Use comparision result only if located within first n characters.
	   %r0: loaded byte count in vreg;
	   %r5: current_len;
	   %r4: n;
	   (current_len - n): [0...16[
	   First ignored match index: loaded bytes - (current_len-n): ]0...16]
	*/
	slgr	%r5,%r4		/* %r5 = current_len - n.  */
	slr	%r0,%r5		/* %r0 = first ignored match index.  */
	vlgvb	%r4,%v18,7	/* Get not equal index or 16 if all equal.  */
	clrjl	%r4,%r0,.Lfound2 /* Jump away if miscompare is within
				     loaded bytes and below n bytes.  */
.Lend_equal:
	lghi	%r2,0
	br	%r14

.Lfound:
	/* Difference or end of string.  */
	/* vfenezf found an unequal element or zero.
	   This instruction compares unsigned words, but wchar_t is signed.
	   Thus we have to compare the found element again.  */
	vlgvb	%r4,%v18,7	/* Extract not equal byte-index.  */
.Lfound2:
	srl	%r4,2		/* And convert it to character-index.  */
	vlgvf	%r0,%v16,0(%r4)	/* Load character-values.  */
	vlgvf	%r1,%v17,0(%r4)
.Lend_cmp_one_char:
	cr	%r0,%r1
	je	.Lend_equal
	lghi	%r2,1
	lghi	%r1,-1
	locgrl	%r2,%r1
	br	%r14
END(WCSNCMP_Z13)

# if ! HAVE_WCSNCMP_IFUNC
strong_alias (WCSNCMP_Z13, wcsncmp)
# endif
#endif
