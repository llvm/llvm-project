/* Vector optimized 32/64 bit S/390 version of wcscmp.
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

#include <ifunc-wcscmp.h>
#if HAVE_WCSCMP_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* int wcscmp (const wchar_t *s1, const wchar_t *s2)
   Compare two strings

   Register usage:
   -r1=loaded byte count s1
   -r2=s1
   -r3=s2
   -r4=loaded byte coutn s2, tmp
   -r5=current_len
   -v16=part of s1
   -v17=part of s2
   -v18=index of unequal
*/
ENTRY(WCSCMP_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

	lghi	%r5,0		/* current_len = 0.  */

.Lloop:
	vlbb	%v16,0(%r5,%r2),6 /* Load s1 to block boundary.  */
	vlbb	%v17,0(%r5,%r3),6 /* Load s2 to block boundary.  */
	lcbb	%r1,0(%r5,%r2),6 /* Get loaded byte count of s1.  */
	jo	.Llt16_1	/* Jump away if vr is not fully loaded.  */
	lcbb	%r4,0(%r5,%r3),6
	jo	.Llt16_2	/* Jump away if vr is not fully loaded.  */
	/* Both vrs are fully loaded.  */
	aghi	%r5,16
	vfenezfs %v18,%v16,%v17	/* Compare not equal with zero search.  */
	jno	.Lfound

	vlbb	%v16,0(%r5,%r2),6
	vlbb	%v17,0(%r5,%r3),6
	lcbb	%r1,0(%r5,%r2),6
	jo	.Llt16_1
	lcbb	%r4,0(%r5,%r3),6
	jo	.Llt16_2
	aghi	%r5,16
	vfenezfs %v18,%v16,%v17
	jno	.Lfound

	vlbb	%v16,0(%r5,%r2),6
	vlbb	%v17,0(%r5,%r3),6
	lcbb	%r1,0(%r5,%r2),6
	jo	.Llt16_1
	lcbb	%r4,0(%r5,%r3),6
	jo	.Llt16_2
	aghi	%r5,16
	vfenezfs %v18,%v16,%v17
	jno	.Lfound

	vlbb	%v16,0(%r5,%r2),6
	vlbb	%v17,0(%r5,%r3),6
	lcbb	%r1,0(%r5,%r2),6
	jo	.Llt16_1
	lcbb	%r4,0(%r5,%r3),6
	jo	.Llt16_2
	aghi	%r5,16
	vfenezfs %v18,%v16,%v17
	jno	.Lfound
	j	.Lloop

.Lcmp_one_char:
	/* At least one of both strings is not 4-byte aligned
	   and there is no full character before next block-boundary.
	   Compare one character to get over the boundary and
	   proceed with normal loop!  */
	vlef	%v16,0(%r5,%r2),0 /* Load one character.  */
	vlef	%v17,0(%r5,%r3),0
	lghi	%r1,4		/* Loaded byte count is 4.  */
	j	.Llt_cmp	/* Proceed with comparision.  */

.Llt16_1:
	lcbb	%r4,0(%r5,%r3),6 /* Get loaded byte count of s2.  */
.Llt16_2:
	clr	%r1,%r4
	locrh	%r1,%r4		/* Get minimum of bytes loaded in s1/2.  */
	nill	%r1,65532	/* Align bytes loaded to full characters.  */
	jz	.Lcmp_one_char	/* Jump away if no full char is available.  */
.Llt_cmp:
	algfr	%r5,%r1		/* Add smallest loaded bytes to current_len.  */
	vfenezfs %v18,%v16,%v17	/* Compare not equal with zero search.  */
	vlgvb	%r4,%v18,7	/* Get not equal index or 16 if all equal.  */
	clrjl	%r4,%r1,.Lfound /* Jump away if miscompare is within loaded
				    bytes.  */
	j	.Lloop

.Lfound:
	/* vfenezf found an unequal element or zero.
	   This instruction compares unsigned words, but wchar_t is signed.
	   Thus we have to compare the found element again.  */
	vlgvb	%r4,%v18,7	/* Extract not equal byte-index,  */
	srl	%r4,2		/* Convert it to character-index.  */
	vlgvf	%r3,%v16,0(%r4)	/* Load character-values.  */
	vlgvf	%r4,%v17,0(%r4)
	cr	%r3,%r4
	je	.Lend_equal
	lghi	%r2,1
	lghi	%r1,-1
	locgrl	%r2,%r1
	br	%r14
.Lend_equal:
	lghi	%r2,0
	br	%r14
END(WCSCMP_Z13)

# if ! HAVE_WCSCMP_IFUNC
strong_alias (WCSCMP_Z13, __wcscmp)
weak_alias (__wcscmp, wcscmp)
# endif

# if ! HAVE_WCSCMP_C && defined SHARED && IS_IN (libc)
strong_alias (WCSCMP_Z13, __GI___wcscmp)
# endif
#endif
