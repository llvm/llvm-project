/* Vector optimized 32/64 bit S/390 version of strchr.
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

#include <ifunc-strchr.h>

#if HAVE_STRCHR_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* char *strchr (const char *s, int c)
   Locate character in string.

   Register usage:
   -r1=tmp
   -r2=s
   -r3=c
   -r4=tmp
   -r5=current_len
   -v16=part of s
   -v17=index of unequal
   -v18=replicated c
*/
ENTRY(STRCHR_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

	vlbb	%v16,0(%r2),6	/* Load s until next 4k-byte boundary.  */
	lcbb	%r1,0(%r2),6	/* Get bytes to 4k-byte boundary or 16.  */

	lghi	%r5,0		/* current_len = 0.  */

	vlvgb	%v18,%r3,0	/* Generate vector which elements are all c.
				   If c > 255, c will be truncated.  */
	vrepb	%v18,%v18,0

	vfeezbs	%v16,%v16,%v18	/* Find element equal with zero search.  */
	vlgvb	%r4,%v16,7	/* Load byte index of character or zero.  */
	clrjl	%r4,%r1,.Lfound /* Return if c/zero is in loaded bytes.  */

	/* Align s to 16 byte.  */
	risbgn	%r4,%r2,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r5,16		/* current_len = 16.  */
	slr	%r5,%r4		/* Compute bytes to 16bytes boundary.  */

	/* Find c/zero in 16 byte aligned loop */
.Lloop:
	vl	%v16,0(%r5,%r2) /* Load s.  */
	vfeezbs	%v16,%v16,%v18	/* Find element equal with zero search.  */
	jno	.Lfound		/* Found c/zero (cc=0|1|2).  */
	vl	%v16,16(%r5,%r2)
	vfeezbs	%v16,%v16,%v18
	jno	.Lfound16
	vl	%v16,32(%r5,%r2)
	vfeezbs	%v16,%v16,%v18
	jno	.Lfound32
	vl	%v16,48(%r5,%r2)
	vfeezbs	%v16,%v16,%v18
	jno	.Lfound48

	aghi	%r5,64
	j	.Lloop		/* No character and no zero -> loop.  */

.Lfound48:
	la	%r5,16(%r5)	/* Use la since aghi would clobber cc.  */
.Lfound32:
	la	%r5,16(%r5)
.Lfound16:
	la	%r5,16(%r5)
.Lfound:
	je	.Lzero		/* Found zero, but no c before that zero.  */

.Lcharacter:
	vlgvb	%r4,%v16,7	/* Load byte index of character.  */
	algr	%r5,%r4
	la	%r2,0(%r5,%r2)	/* Return pointer to character.  */
	br	%r14

.Lzero:
	llgcr	%r3,%r3		/* char c_char = (char) c.  */
	clije	%r3,0,.Lcharacter /* Found zero and c is zero.  */
	lghi	%r2,0		/* Return null if character not found.  */
	br	%r14
END(STRCHR_Z13)

# if ! HAVE_STRCHR_IFUNC
strong_alias (STRCHR_Z13, strchr)
weak_alias (strchr, index)
# endif

# if ! HAVE_STRCHR_C && defined SHARED && IS_IN (libc)
strong_alias (STRCHR_Z13, __GI_strchr)
# endif

#endif /* HAVE_STRCHR_Z13  */
