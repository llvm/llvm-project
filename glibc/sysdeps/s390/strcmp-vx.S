/* Vector optimized 32/64 bit S/390 version of strcmp.
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

#include <ifunc-strcmp.h>
#if HAVE_STRCMP_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* int strcmp (const char *s1, const char *s2)
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
ENTRY(STRCMP_Z13)
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
	vfenezbs %v18,%v16,%v17	/* Compare not equal with zero search.  */
	jno	.Lfound

	vlbb	%v16,0(%r5,%r2),6
	vlbb	%v17,0(%r5,%r3),6
	lcbb	%r1,0(%r5,%r2),6
	jo	.Llt16_1
	lcbb	%r4,0(%r5,%r3),6
	jo	.Llt16_2
	aghi	%r5,16
	vfenezbs %v18,%v16,%v17
	jno	.Lfound

	vlbb	%v16,0(%r5,%r2),6
	vlbb	%v17,0(%r5,%r3),6
	lcbb	%r1,0(%r5,%r2),6
	jo	.Llt16_1
	lcbb	%r4,0(%r5,%r3),6
	jo	.Llt16_2
	aghi	%r5,16
	vfenezbs %v18,%v16,%v17
	jno	.Lfound

	vlbb	%v16,0(%r5,%r2),6
	vlbb	%v17,0(%r5,%r3),6
	lcbb	%r1,0(%r5,%r2),6
	jo	.Llt16_1
	lcbb	%r4,0(%r5,%r3),6
	jo	.Llt16_2
	aghi	%r5,16
	vfenezbs %v18,%v16,%v17
	jno	.Lfound
	j	.Lloop

.Llt16_1:
	lcbb	%r4,0(%r5,%r3),6 /* Get loaded byte count of s2.  */
.Llt16_2:
	clr	%r1,%r4
	locrh	%r1,%r4		/* Get minimum of bytes loaded in s1/2.  */
	algfr	%r5,%r1		/* Add smallest loaded bytes to current_len.  */
	vfenezbs %v18,%v16,%v17	/* Compare not equal with zero search.  */
	vlgvb	%r4,%v18,7	/* Get not equal index or 16 if all equal.  */
	clrjl	%r4,%r1,.Lfound /* Jump away if miscompare is within loaded
				    bytes.  */
	j	.Lloop

.Lfound:
	je	.Lend_equal
	lghi	%r2,1
	lghi	%r1,-1
	locgrl	%r2,%r1
	br	%r14
.Lend_equal:
	lghi	%r2,0
	br	%r14
END(STRCMP_Z13)

# if ! HAVE_STRCMP_IFUNC
strong_alias (STRCMP_Z13, strcmp)
# endif

# if ! HAVE_STRCMP_Z900_G5 && defined SHARED && IS_IN (libc)
strong_alias (STRCMP_Z13, __GI_strcmp)
# endif
#endif
