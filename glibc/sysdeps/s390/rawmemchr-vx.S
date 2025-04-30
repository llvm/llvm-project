/* Vector optimized 32/64 bit S/390 version of rawmemchr.
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

#include <ifunc-rawmemchr.h>

#if HAVE_RAWMEMCHR_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* void *rawmemchr (const void *s, int c)
   Scans memory for character c
   and returns pointer to first c.

   Register usage:
   -r1=tmp
   -r2=s
   -r3=c
   -r4=tmp
   -r5=current_len
   -v16=part of s
   -v17=index of unequal
   -v18=c replicated
*/
ENTRY(RAWMEMCHR_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

	vlbb	%v16,0(%r2),6	/* Load s until next 4k-byte boundary.  */
	lcbb	%r1,0(%r2),6	/* Get bytes to 4k-byte boundary or 16.  */

	vlvgb	%v18,%r3,0	/* Generate vector which elements are all c.
				   If c > 255, c will be truncated.  */
	vrepb	%v18,%v18,0

	vfeeb	%v17,%v16,%v18	/* Vector find element equal.  */
	vlgvb	%r5,%v17,7	/* Load byte index of character or zero.  */
	clrjl	%r5,%r1,.Lend_found /* If found c is in loaded bytes, end.  */

	/* Align s to 16 byte.  */
	risbgn	%r1,%r2,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r5,16
	slr	%r5,%r1		/* Compute bytes to 16bytes boundary.  */

	/* Find c in a 16byte aligned loop.  */
.Lloop:
	vl	%v16,0(%r5,%r2)	/* Load s.  */
	vfeebs	%v17,%v16,%v18	/* Vector find element equal.  */
	jno	.Lcharacter	/* Jump away if element found.  */
	vl	%v16,16(%r5,%r2)
	vfeebs	%v17,%v16,%v18
	jno	.Lcharacter16
	vl	%v16,32(%r5,%r2)
	vfeebs	%v17,%v16,%v18
	jno	.Lcharacter32
	vl	%v16,48(%r5,%r2)
	vfeebs	%v17,%v16,%v18
	jno	.Lcharacter48

	aghi	%r5,64
	j	.Lloop		/* No character found -> loop.  */

	/* Found character.  */
.Lcharacter48:
	aghi %r5,16
.Lcharacter32:
	aghi %r5,16
.Lcharacter16:
	aghi %r5,16
.Lcharacter:
	vlgvb	%r1,%v17,7	/* Load byte index of character.  */
	algr	%r5,%r1
.Lend_found:
	la	%r2,0(%r5,%r2)	/* Return pointer to character.  */
	br	%r14
END(RAWMEMCHR_Z13)

# if ! HAVE_RAWMEMCHR_IFUNC
strong_alias (RAWMEMCHR_Z13, __rawmemchr)
weak_alias (__rawmemchr, rawmemchr)
# endif

# if ! HAVE_RAWMEMCHR_C && defined SHARED && IS_IN (libc)
strong_alias (RAWMEMCHR_Z13, __GI___rawmemchr)
# endif

#endif /* HAVE_RAWMEMCHR_Z13  */
