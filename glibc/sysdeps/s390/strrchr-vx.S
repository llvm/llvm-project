/* Vector optimized 32/64 bit S/390 version of strrchr.
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

#include <ifunc-strrchr.h>

#if HAVE_STRRCHR_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* char *strrchr (const char *s, int c)
   Locate the last character c in string.

   Register usage:
   -r0=loaded bytes in first part of s.
   -r1=pointer to last occurence of c or NULL if not found.
   -r2=s
   -r3=c
   -r4=tmp
   -r5=current_len
   -v16=part of s
   -v17=index of found element
   -v18=replicated c
   -v19=part of s with last occurence of c.
   -v20=permute pattern
*/
ENTRY(STRRCHR_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

	vlbb	%v16,0(%r2),6	/* Load s until next 4k-byte boundary.  */
	lcbb	%r0,0(%r2),6	/* Get bytes to 4k-byte boundary or 16.  */

	vlvgb	%v18,%r3,0	/* Generate vector which elements are all c.
				   if c > 255, c will be truncated.  */
	vrepb	%v18,%v18,0

	lghi	%r1,-1		/* Currently no c found.  */
	lghi	%r5,0		/* current_len = 0.  */

	vfeezbs	%v17,%v16,%v18	/* Find element equal or zero.  */
	vlgvb	%r4,%v17,7	/* Load byte index of c/zero or 16.  */
	clrjl	%r4,%r0,.Lfound_first_part /* Found c/zero in loaded bytes.  */
.Lalign:
	/* Align s to 16 byte.  */
	risbgn	%r4,%r2,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r5,16		/* current_len = 16.  */
	slr	%r5,%r4		/* Compute bytes to 16bytes boundary.  */

.Lloop:
	vl	%v16,0(%r5,%r2) /* Load s.  */
	vfeezbs	%v17,%v16,%v18	/* Find element equal with zero search.  */
	jno	.Lfound		/* Found c/zero (cc=0|1|2).  */
	vl	%v16,16(%r5,%r2)
	vfeezbs	%v17,%v16,%v18
	jno	.Lfound16
	vl	%v16,32(%r5,%r2)
	vfeezbs	%v17,%v16,%v18
	jno	.Lfound32
	vl	%v16,48(%r5,%r2)
	vfeezbs	%v17,%v16,%v18
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
	/* Save this part of s to check for further matches after reaching
	   the end of the complete string.  */
	vlr	%v19,%v16
	lgr	%r1,%r5

	jh	.Lzero		/* Found a zero after the found c.  */
	aghi	%r5,16		/* Start search of next part of s.  */
	j	.Lloop

.Lfound_first_part:
	/* This code is only executed if the found c/zero is whithin loaded
	   bytes. If no c/zero was found (cc==3) the found index = 16, thus
	   this code is not called.
	   Resulting condition code of vector find element equal:
	   cc==0: no c, found zero
	   cc==1: c found, no zero
	   cc==2: c found, found zero after c
	   cc==3: no c, no zero (this case can be ignored).  */
	je	.Lzero		/* Found zero, but no c before that zero.  */

	locgrne	%r1,%r5		/* Mark c as found in first part of s.  */
	vlr	%v19,%v16

	jl	.Lalign		/* No zero (e.g. if vr was fully loaded)
				   -> Align and loop afterwards.  */

	/* Found a zero in vr. If vr was not fully loaded due to block
	   boundary, the remaining bytes are filled with zero and we can't
	   rely on zero indication of condition code here!  */

	vfenezb	%v17,%v16,%v16	/* Find zero.  */
	vlgvb	%r4,%v17,7	/* Load byte index of zero or 16.  */
	clrjl	%r4,%r0,.Lzero	/* Zero within loaded bytes -> end.  */
	j	.Lalign		/* Align and loop afterwards.  */

.Lend_searched_zero:
	vlgvb	%r4,%v17,7	/* Load byte index of zero.  */
	algr	%r5,%r4
	la	%r2,0(%r5,%r2)	/* Return pointer to zero.  */
	br	%r14

.Lzero:
	/* Reached end of string. Check if one c was found before.  */
	clije	%r3,0,.Lend_searched_zero /* Found zero and c is zero.  */

	cgfi	%r1,-1		/* No c found -> return NULL.  */
	locghie	%r2,0
	ber	%r14

	larl	%r3,.Lpermute_mask /* Load permute mask.  */
	vl	%v20,0(%r3)

	/* c was found and is part of v19.  */
	vfenezb	%v17,%v19,%v19	/* Find zero.  */
	vlgvb	%r4,%v17,7	/* Load byte index of zero or 16.  */

	clgfi	%r5,0		/* Loaded byte count in v19 is 16, ...  */
	lochine	%r0,16		/* ... if v19 is not the first part of s.  */
	ahi	%r0,-1		/* Convert byte count to highest index.  */

	clr	%r0,%r4
	locrl	%r4,%r0		/* r4 = min (zero-index, highest-index).  */

	/* Right-shift of v19 to mask bytes after zero.  */
	clije	%r4,15,.Lzero_permute /* No shift is needed if highest index
					 in vr is 15.  */
	lhi	%r0,15
	slr	%r0,%r4		/* Compute byte count for vector shift right.  */
	sll	%r0,3		/* Convert to bit count.  */
	vlvgb	%v17,%r0,7
	vsrlb	%v19,%v19,%v17	/* Vector shift right by byte by number of bytes
				   specified in bits 1-4 of byte 7 in v17.   */

	/* Reverse bytes in v19.  */
.Lzero_permute:
	vperm	%v19,%v19,%v19,%v20 /* Permute v19 to reversed order.  */

	/* Find c in reversed v19.  */
	vfeeb	%v19,%v19,%v18	/* Find c.  */
	la	%r2,0(%r1,%r2)
	vlgvb	%r3,%v19,7	/* Load byte index of c.  */

	/* Compute index in real s and return.  */
	slgr	%r4,%r3
	la	%r2,0(%r4,%r2)	/* Return pointer to zero.  */
	br	%r14
.Lpermute_mask:
	.byte	0x0F,0x0E,0x0D,0x0C,0x0B,0x0A,0x09,0x08
	.byte	0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00
END(STRRCHR_Z13)

# if ! HAVE_STRRCHR_IFUNC
strong_alias (STRRCHR_Z13, strrchr)
weak_alias (strrchr, rindex)
# endif

# if ! HAVE_STRRCHR_C && defined SHARED && IS_IN (libc)
strong_alias (STRRCHR_Z13, __GI_strrchr)
# endif

#endif /* HAVE_STRRCHR_Z13  */
