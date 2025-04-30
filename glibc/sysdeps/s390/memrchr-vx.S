/* Vector optimized 32/64 bit S/390 version of memrchr.
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

#include <ifunc-memrchr.h>

#if HAVE_MEMRCHR_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* void *memrchr (const void *s, int c, size_t n)
   Scans memory for character c backwards
   and returns pointer to first c.

   Register usage:
   -r0=tmp
   -r1=tmp
   -r2=s
   -r3=c
   -r4=n
   -r5=s in loop

   -v16=part of s
   -v17=index of found c
   -v18=c replicated
   -v20=permute pattern
*/
ENTRY(MEMRCHR_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

# if !defined __s390x__
	llgfr	%r4,%r4
# endif /* !defined __s390x__ */
	clgije	%r4,0,.Lnot_found

	vlvgb	%v18,%r3,0	/* Generate vector which elements are all c.
				   If c > 255, c will be truncated.  */
	vrepb	%v18,%v18,0

	llcr	%r3,%r3		/* char c_char = (char) c.  */

	/* check byte n - 1.  */
	llc	%r0,-1(%r4,%r2)
	slgfi	%r4,1
	clrje	%r0,%r3,.Lfound_end
	jh	.Lnot_found	/* Return NULL if n is now 0.  */

	larl	%r1,.Lpermute_mask /* Load permute mask.  */
	vl	%v20,0(%r1)

	/* check byte n - 2.  */
	llc	%r0,-1(%r4,%r2)
	slgfi	%r4,1
	clrje	%r0,%r3,.Lfound_end
	jh	.Lnot_found	/* Return NULL if n is now 0.  */

	clgijhe	%r4,64,.Lloop64	/* If n >= 64 -> loop64.  */

.Llt64:
	/* Process n < 64 bytes.  */
	clgijl	%r4,16,.Llt16	/* Jump away if n < 16.  */
	aghi	%r4,-16
	vl	%v16,0(%r4,%r2)
	vfeebs	%v17,%v16,%v18
	jno	.Lfound0
	clgijl	%r4,16,.Llt16
	aghi	%r4,-16
	vl	%v16,0(%r4,%r2)
	vfeebs	%v17,%v16,%v18
	jno	.Lfound0
	clgijl	%r4,16,.Llt16
	aghi	%r4,-16
	vl	%v16,0(%r4,%r2)
	vfeebs	%v17,%v16,%v18
	jno	.Lfound0
.Llt16:
	clgfi	%r4,0		/* if remaining bytes == 0, return NULL.  */
	locghie	%r2,0
	ber	%r14

	aghi	%r4,-1		/* vll needs highest index.  */
	vll	%v16,%r4,0(%r2)	/* Load remaining bytes.  */

	/* Right-shift of v16 to mask bytes after highest index.  */
	lhi	%r0,15
	slr	%r0,%r4		/* Compute byte count for vector shift right.  */
	sll	%r0,3		/* Convert to bit count.  */
	vlvgb	%v17,%r0,7
	vsrlb	%v16,%v16,%v17	/* Vector shift right by byte by number of bytes
				   specified in bits 1-4 of byte 7 in v17.   */
	j	.Lfound_permute

.Lfound48:
	aghi	%r4,16
.Lfound32:
	aghi	%r4,16
.Lfound16:
	aghi	%r4,16
.Lfound0:
	la	%r2,0(%r4,%r2)	/* Set pointer to start of v16.  */
	lghi	%r4,15		/* Set highest index in v16 to last index.  */
.Lfound_permute:
	/* Search for a c in v16 in reversed byte order. v16 contains %r4 + 1
	   bytes. If v16 was not fully loaded, the bytes are already
	   right shifted, so that the bytes in v16 can simply be reversed.  */
	vperm	%v16,%v16,%v16,%v20 /* Permute v16 to reversed order.  */
	vfeeb	%v16,%v16,%v18	/* Find c in reversed v16.  */
	vlgvb	%r1,%v16,7	/* Index of c or 16 if not found.  */

	/* Return NULL if there is no c in loaded bytes. */
	clrjh	%r1,%r4,.Lnot_found

	slgr	%r4,%r1
.Lfound_end:
	la	%r2,0(%r4,%r2)	/* Return pointer to c.  */
	br	%r14

.Lnot_found:
	lghi	%r2,0
	br	%r14

.Lpermute_mask:
	.byte	0x0F,0x0E,0x0D,0x0C,0x0B,0x0A,0x09,0x08
	.byte	0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00

.Lloop64:
	aghi	%r4,-64
	vl	%v16,48(%r4,%r2) /* Load 16bytes of memory area.  */
	vfeebs	%v17,%v16,%v18	/* Find c.  */
	jno	.Lfound48	/* Jump away if c was found.  */
	vl	%v16,32(%r4,%r2)
	vfeebs	%v17,%v16,%v18
	jno	.Lfound32
	vl	%v16,16(%r4,%r2)
	vfeebs	%v17,%v16,%v18
	jno	.Lfound16
	vl	%v16,0(%r4,%r2)
	vfeebs	%v17,%v16,%v18
	jno	.Lfound0

	clgijhe	%r4,64,.Lloop64	/* If n >= 64 -> loop64.  */
	j	.Llt64
END(MEMRCHR_Z13)

# if ! HAVE_MEMRCHR_IFUNC
strong_alias (MEMRCHR_Z13, __memrchr)
weak_alias (__memrchr, memrchr)
# endif

#endif /* HAVE_MEMRCHR_Z13  */
