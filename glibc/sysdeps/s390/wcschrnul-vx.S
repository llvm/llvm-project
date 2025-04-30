/* Vector optimized 32/64 bit S/390 version of wcschrnul.
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

#include <ifunc-wcschrnul.h>
#if HAVE_WCSCHRNUL_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* wchar_t* wcschrnul (const wchar_t *s, wchar_t c)
   Returns pointer to first c or to \0 if c not found.

   Register usage:
   -r1=tmp
   -r2=s and return pointer
   -r3=c
   -r4=tmp
   -r5=current_len
   -v16=part of s
   -v18=vector with c replicated in every byte
*/
ENTRY(WCSCHRNUL_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

	vlbb	%v16,0(%r2),6	/* Load s until next 4k-byte boundary.  */
	lcbb	%r1,0(%r2),6	/* Get bytes to 4k-byte boundary or 16.  */

	tmll	%r2,3		/* Test if s is 4-byte aligned?   */
	jne	.Lfallback	/* And use common-code variant if not.  */

	lghi	%r5,0		/* current_len = 0.  */

	vlvgf	%v18,%r3,0	/* Generate vector which elements are all c.  */
	vrepf	%v18,%v18,0

	vfeezfs	%v16,%v16,%v18	/* Find element equal with zero search.  */
	vlgvb	%r4,%v16,7	/* Load byte index of character or zero.  */
	clrjl	%r4,%r1,.Lfound /* Return if c/zero is in loaded bytes.  */

	/* Align s to 16 byte.  */
	risbgn	%r4,%r2,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r5,16		/* current_len = 16.  */
	slr	%r5,%r4		/* Compute bytes to 16bytes boundary.  */

	/* Find c/zero in 16byte aligned loop */
.Lloop:
	vl	%v16,0(%r5,%r2) /* Load s.  */
	vfeezfs	%v16,%v16,%v18	/* Find element equal with zero search.  */
	jno	.Lfound		/* Found c/zero (cc=0|1|2).  */
	vl	%v16,16(%r5,%r2)
	vfeezfs	%v16,%v16,%v18
	jno	.Lfound16
	vl	%v16,32(%r5,%r2)
	vfeezfs	%v16,%v16,%v18
	jno	.Lfound32
	vl	%v16,48(%r5,%r2)
	vfeezfs	%v16,%v16,%v18
	jno	.Lfound48

	aghi	%r5,64
	j	.Lloop		/* No character and no zero -> loop.  */

	/* Found character or zero */
.Lfound48:
	aghi	%r5,16
.Lfound32:
	aghi	%r5,16
.Lfound16:
	aghi	%r5,16
.Lfound:
	vlgvb	%r1,%v16,7	/* Load byte index of character.  */
	algr	%r5,%r1
	la	%r2,0(%r5,%r2)	/* Return pointer to character.  */

.Lend:
	br	%r14
.Lfallback:
	jg	WCSCHRNUL_C
END(WCSCHRNUL_Z13)

# if ! HAVE_WCSCHRNUL_IFUNC
strong_alias (WCSCHRNUL_Z13, __wcschrnul)
weak_alias (__wcschrnul, wcschrnul)
# endif
#endif
