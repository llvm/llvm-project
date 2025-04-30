/* Vector optimized 32/64 bit S/390 version of memccpy.
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

#include <ifunc-memccpy.h>

#if HAVE_MEMCCPY_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* void *memccpy (void * dest, const void *src, int c, size_t n)
   Copies no more than n bytes from src to dest,
   stopping when the character c is found
   and returns pointer next to c in dest or null if c not found.

   Register usage:
   -r0=tmp
   -r1=tmp
   -r2=dest
   -r3=src
   -r4=c
   -r5=n
   -r6=current_len
   -v16=part of s
   -v17=index of found c
   -v18=c replicated
   -v19=part #2 of s
   -v31=save area for r6
*/
ENTRY(MEMCCPY_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

# if !defined __s390x__
	llgfr	%r5,%r5
# endif /* !defined __s390x__ */

	vlvgp	%v31,%r6,%r7	/* Save registers.  */
	clgije	%r5,0,.Lnf_end	/* If len == 0 then exit.  */

	vlbb	%v16,0(%r3),6	/* Load s until next 4k-byte boundary.  */
	lcbb	%r0,0(%r3),6	/* Get bytes to 4k-byte boundary or 16.  */
	llgfr	%r0,%r0		/* Convert 32bit to 64bit.  */

	vlvgb	%v18,%r4,0	/* Generate vector which elements are all c.
				   if c > 255, c will be truncated.  */
	vrepb	%v18,%v18,0
	lghi	%r6,0		/* current_len = 0.  */

	clgrjle	%r5,%r0,.Lremaining_v16 /* If maxlen <= loaded-bytes
					   -> Process remaining.  */

	vfeebs	%v17,%v16,%v18	/* Find c.  */
	vlgvb	%r1,%v17,7	/* Load byte index of c.  */
	clgrjl	%r1,%r0,.Lfound_v16 /* Found c is within loaded bytes.  */

	/* Align s to 16 byte.  */
	risbgn	%r1,%r3,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r6,15		/* current_len = 15.  */
	slr	%r6,%r1		/* Compute highest index to 16byte boundary.  */

	vstl	%v16,%r6,0(%r2)	/* Store prcessed bytes */
	ahi	%r6,1

.Lpreloop1:
	/* Now we are 16byte aligned, so we can load
	   a full vreg without page fault.  */
	vl	%v16,0(%r6,%r3)	/* Load s.  */
	clgijl	%r5,17,.Lremaining_v16	/* If n <= 16,
					   process remaining bytes.  */
	lgr	%r7,%r5
	slgfi	%r7,16		/* border_len = n - 16.  */
	j	.Lloop1

.Lloop2:
	vl	%v16,16(%r6,%r3)
	vst	%v19,0(%r6,%r2)
	aghi	%r6,16

.Lloop1:
	clgrjhe	%r6,%r7,.Lremaining_v16 /* If current_len >= border
					   then process remaining bytes.  */
	vfeebs	%v17,%v16,%v18	/* Find c.  */
	jl	.Lfound_v16	/* Jump away if c was found.  */
	vl	%v19,16(%r6,%r3) /* Load next s part.  */
	vst	%v16,0(%r6,%r2)	/* Store previous part without c.  */
	aghi	%r6,16

	clgrjhe	%r6,%r7,.Lremaining_v19
	vfeebs	%v17,%v19,%v18
	jl	.Lfound_v19
	vl	%v16,16(%r6,%r3)
	vst	%v19,0(%r6,%r2)
	aghi	%r6,16

	clgrjhe	%r6,%r7,.Lremaining_v16
	vfeebs	%v17,%v16,%v18
	jl	.Lfound_v16
	vl	%v19,16(%r6,%r3)
	vst	%v16,0(%r6,%r2)
	aghi	%r6,16

	clgrjhe	%r6,%r7,.Lremaining_v19
	vfeebs	%v17,%v19,%v18
	jo	.Lloop2

.Lfound_v19:
	vlr	%v16,%v19
.Lfound_v16:
	/* v16 contains c. Store remaining bytes to c. currlen hasn´t
	   reached border, thus checking for maxlen is not needed! */
	vlgvb	%r1,%v17,7	/* Load byte index of c.  */
	la	%r2,0(%r6,%r2)	/* vstl has no support for index-register.  */
.Lfound_v16_store:
	vstl	%v16,%r1,0(%r2)	/* Copy bytes including c.  */
	la	%r2,1(%r1,%r2)	/* Return pointer next to c in dest.  */
	vlgvg	%r6,%v31,0
	vlgvg	%r7,%v31,1
	br	%r14

.Lremaining_v19:
	vlr	%v16,%v19
.Lremaining_v16:
	/* v16 contains the remaining bytes [1...16].
	   Check and store remaining bytes.  */
	vfeebs	%v17,%v16,%v18
	slgrk	%r7,%r5,%r6	/* Remaining bytes = maxlen - current_len.  */
	aghi	%r7,-1		/* vstl needs highest index.  */
	la	%r2,0(%r6,%r2)	/* vstl has no index register.  */
	vlgvb	%r1,%v17,7	/* Load index of c or 16 if not found.  */
	/* c in remaining bytes? -> Jump away (c-index <= max-index)  */
	clrjle	%r1,%r7,.Lfound_v16_store
	vstl	%v16,%r7,0(%r2)	/* Store remaining bytes.  */

.Lnf_end:
	vlgvg	%r6,%v31,0
	vlgvg	%r7,%v31,1
	lghi	%r2,0		/* Return null.  */
	br	%r14
END(MEMCCPY_Z13)

# if ! HAVE_MEMCCPY_IFUNC
strong_alias (MEMCCPY_Z13, __memccpy)
weak_alias (__memccpy, memccpy)
# endif

#endif /* HAVE_MEMCCPY_Z13  */
