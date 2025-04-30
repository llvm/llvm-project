/* Vector optimized 32/64 bit S/390 version of strncmp.
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

#include <ifunc-strncmp.h>

#if HAVE_STRNCMP_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* int strncmp (const char *s1, const char *s2, size_t n)
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
ENTRY(STRNCMP_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

# if !defined __s390x__
	llgfr	%r4,%r4
# endif /* !defined __s390x__ */

	clgije	%r4,0,.Lend_equal /* Nothing to do if n == 0,  */
	lghi	%r5,0		/* current_len = 0.  */

.Lloop:
	vlbb	%v16,0(%r5,%r2),6 /* Load s1 to block boundary.  */
	vlbb	%v17,0(%r5,%r3),6 /* Load s2 to block boundary.  */
	lcbb	%r0,0(%r5,%r2),6 /* Get loaded byte count of s1.  */
	jo	.Llt16_1	/* Jump away if vr is not fully loaded.  */
	lcbb	%r1,0(%r5,%r3),6 /* Get loaded byte count of s2.  */
	jo	.Llt16_2	/* Jump away if vr is not fully loaded.  */
	aghi	%r5,16		/* Both vrs are fully loaded.  */
	clgrjhe	%r5,%r4,.Llastcmp /* If current_len >= n ->last compare.  */
	vfenezbs %v18,%v16,%v17	/* Compare not equal with zero search.  */
	jno	.Lfound

	vlbb	%v16,0(%r5,%r2),6
	vlbb	%v17,0(%r5,%r3),6
	lcbb	%r0,0(%r5,%r2),6
	jo	.Llt16_1
	lcbb	%r1,0(%r5,%r3),6
	jo	.Llt16_2
	aghi	%r5,16
	clgrjhe	%r5,%r4,.Llastcmp
	vfenezbs %v18,%v16,%v17
	jno	.Lfound

	vlbb	%v16,0(%r5,%r2),6
	vlbb	%v17,0(%r5,%r3),6
	lcbb	%r0,0(%r5,%r2),6
	jo	.Llt16_1
	lcbb	%r1,0(%r5,%r3),6
	jo	.Llt16_2
	aghi	%r5,16
	clgrjhe	%r5,%r4,.Llastcmp
	vfenezbs %v18,%v16,%v17
	jno	.Lfound

	vlbb	%v16,0(%r5,%r2),6
	vlbb	%v17,0(%r5,%r3),6
	lcbb	%r0,0(%r5,%r2),6
	jo	.Llt16_1
	lcbb	%r1,0(%r5,%r3),6
	jo	.Llt16_2
	aghi	%r5,16
	clgrjhe	%r5,%r4,.Llastcmp
	vfenezbs %v18,%v16,%v17
	jno	.Lfound
	j	.Lloop

.Llt16_1:
	lcbb	%r1,0(%r5,%r3),6 /* Get loaded byte count ofs2.  */
.Llt16_2:
	clr	%r0,%r1		/* Compare logical.  */
	locrh	%r0,%r1		/* Compute minimum of bytes loaded.  */
	algfr	%r5,%r0		/* Add smallest loaded bytes to current_len.  */
	clgrj	%r5,%r4,10,.Llastcmp /* If current_len >= n ->last compare.  */
	vfenezbs %v18,%v16,%v17	/* Compare not equal with zero search.  */
	vlgvb	%r1,%v18,7	/* Get not equal index or 16 if all equal.  */
	clrjl	%r1,%r0,.Lfound /* Jump away if miscompare is within
				    loaded bytes (index < loaded-bytes) */
	j	.Lloop

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
	vfenezbs %v18,%v16,%v17	/* Compare not equal with zero search.  */
	vlgvb	%r1,%v18,7	/* Get not equal index or 16 if all equal.  */
	clrjl	%r1,%r0,.Lfound /* Jump away if miscompare is within
				    loaded bytes and below n bytes.  */
	j	.Lend_equal	/* Miscompare after n-bytes -> end equal.  */

.Lfound:
	/* Difference or end of string.  */
	je	.Lend_equal
	lghi	%r2,1
	lghi	%r1,-1
	locgrl	%r2,%r1
	br	%r14
.Lend_equal:
	lghi	%r2,0
	br	%r14
END(STRNCMP_Z13)

# if ! HAVE_STRNCMP_IFUNC
strong_alias (STRNCMP_Z13, strncmp)
# endif

# if ! HAVE_STRNCMP_C && defined SHARED && IS_IN (libc)
strong_alias (STRNCMP_Z13, __GI_strncmp)
# endif

#endif /* HAVE_STRNCMP_Z13  */
