/* Vector optimized 32/64 bit S/390 version of strncpy.
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

#include <ifunc-strncpy.h>
#include "sysdep.h"
#include "asm-syntax.h"

	.text

#if HAVE_STRNCPY_Z13
/* char * strncpy (const char *dest, const char *src, size_t n)
   Copy at most n characters of string  src to dest.

   Register usage:
   -r0=dest pointer for return
   -r1=tmp, zero byte index
   -r2=dest
   -r3=src
   -r4=n
   -r5=current_len
   -r6=tmp, loaded bytes
   -r7=tmp, border
   -v16=part of src
   -v17=index of zero
   -v18=part of src
   -v31=register save area for r6, r7
*/
ENTRY(STRNCPY_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

# if !defined __s390x__
	llgfr	%r4,%r4
# endif /* !defined __s390x__ */

	clgfi	%r4,0
	ber	%r14		/* Nothing to do, if n == 0.  */
	lgr	%r0,%r2		/* Save destination pointer for return.  */
	vlvgp	%v31,%r6,%r7	/* Save registers.  */

	vlbb	%v16,0(%r3),6	/* Load s until next 4k-byte boundary.  */
	lcbb	%r6,0(%r3),6	/* Get bytes to 4k-byte boundary or 16.  */
	llgfr	%r6,%r6		/* Convert 32bit to 64bit.  */

	lghi	%r5,0		/* current_len = 0.  */

	clgrjle	%r4,%r6,.Lremaining_v16 /* If n <= loaded-bytes
					   -> process remaining.  */

	/* n > loaded-byte-count.  */
	vfenezb	%v17,%v16,%v16	/* Find element not equal with zero search.  */
	vlgvb	%r1,%v17,7	/* Load zero index or 16 if not found.  */
	clrjl	%r1,%r6,.Lfound_v16_store /* Found zero within loaded bytes,
					     copy and return.  */

	/* Align s to 16 byte.  */
	risbgn	%r7,%r3,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r5,15		/* current_len = 15.  */
	slr	%r5,%r7		/* Compute highest index to 16byte boundary.  */

	/* Zero not found and n > loaded-byte-count.  */
	vstl	%v16,%r5,0(%r2)	/* Copy loaded characters - no zero.  */
	ahi	%r5,1		/* Start loop at next character.  */

	/* Now we are 16byte aligned, so we can load
	   a full vreg without page fault.  */
	lgr	%r1,%r5		/* If %r5 + 64 < maxlen? -> loop64.  */
	aghi	%r1,64
	clgrjl	%r1,%r4,.Lloop64

	vl	%v16,0(%r5,%r3)	/* Load s.  */
	clgijl	%r4,17,.Lremaining_v16	/* If n <= 16, process remaining
					   bytes.  */
.Llt64:
	lgr	%r7,%r4
	slgfi	%r7,16		/* border_len = n - 16.  */

	clgrjhe	%r5,%r7,.Lremaining_v16 /* If current_len >= border
					   then process remaining bytes.  */
	vfenezbs %v17,%v16,%v16	/* Find element not equal with zero search.  */
	je	.Lfound_v16	/* Jump away if zero was found.  */
	vl	%v18,16(%r5,%r3) /* Load next part of s.  */
	vst	%v16,0(%r5,%r2)	/* Store previous part without zero to dst.  */
	aghi	%r5,16

	clgrjhe	%r5,%r7,.Lremaining_v18
	vfenezbs %v17,%v18,%v18
	je	.Lfound_v18
	vl	%v16,16(%r5,%r3)
	vst	%v18,0(%r5,%r2)
	aghi	%r5,16

	clgrjhe	%r5,%r7,.Lremaining_v16
	vfenezbs %v17,%v16,%v16
	je	.Lfound_v16
	vl	%v18,16(%r5,%r3)
	vst	%v16,0(%r5,%r2)
	aghi	%r5,16

.Lremaining_v18:
	vlr	%v16,%v18
.Lremaining_v16:
	/* v16 contains the remaining bytes [1...16].
	   Store remaining bytes and append string-termination.  */
	vfenezb	%v17,%v16,%v16	/* Find element not equal with zero search.  */
	slgrk	%r7,%r4,%r5	/* Remaining bytes = maxlen - current_len.  */
	aghi	%r7,-1		/* vstl needs highest index.  */
	la	%r2,0(%r5,%r2)	/* vstl has no index register.  */
	vlgvb	%r1,%v17,7	/* Load zero index or 16 if not found.  */
	/* Zero in remaining bytes? -> jump away (zero-index < max-index)
	   Do not jump away if zero-index == max-index,
	   but simply copy zero with vstl below.  */
	clrjl	%r1,%r7,.Lfound_v16_store
	vstl	%v16,%r7,0(%r2)	/* Store remaining bytes without null
				   termination!.  */
.Lend:
	/* Restore saved registers.  */
	vlgvg	%r6,%v31,0
	vlgvg	%r7,%v31,1
	lgr	%r2,%r0		/* Load saved dest-ptr.  */
	br	%r14


.Lfound_v16_32:
	aghi	%r5,32
	j	.Lfound_v16
.Lfound_v18_48:
	aghi	%r5,32
.Lfound_v18_16:
	aghi	%r5,16
.Lfound_v18:
	vlr	%v16,%v18
.Lfound_v16:
	/* v16 contains a zero. Store remaining bytes to zero. current_len
	   has not reached border, thus checking for n is not needed! */
	vlgvb	%r1,%v17,7	/* Load byte index of zero.  */
	la	%r2,0(%r5,%r2)	/* vstl has no support for index-register.  */
.Lfound_v16_store:
	vstl	%v16,%r1,0(%r2)	/* Copy characters including zero.  */
	/* Fill remaining bytes with zero - remaining count always > 0.  */
	algr	%r5,%r1		/* Remaining bytes (=%r4) = ...  */
	slgr	%r4,%r5		/* = n - (current_len + zero_index + 1).  */
	la	%r2,0(%r1,%r2)	/* Pointer to zero. start filling beyond.  */
	aghi	%r4,-2		/* mvc with exrl needs count - 1.
				   (additional -1, see remaining bytes above) */
	srlg	%r6,%r4,8	/* Split into 256 byte blocks.  */
	ltgr	%r6,%r6
	je	.Lzero_lt256
.Lzero_loop256:
	mvc	1(256,%r2),0(%r2) /* Fill 256 zeros at once.  */
	la	%r2,256(%r2)
	brctg	%r6,.Lzero_loop256 /* Loop until all blocks are processed.  */
.Lzero_lt256:
	exrl	%r4,.Lmvc_lt256
	j	.Lend
.Lmvc_lt256:
	mvc	1(1,%r2),0(%r2)

.Lloop64:
	vl	%v16,0(%r5,%r3)	/* Load s.  */
	vfenezbs %v17,%v16,%v16	/* Find element not equal with zero search.  */
	je	.Lfound_v16	/* Jump away if zero was found.  */
	vl	%v18,16(%r5,%r3) /* Load next part of s.  */
	vst	%v16,0(%r5,%r2)	/* Store previous part without zero to dst.  */
	vfenezbs %v17,%v18,%v18
	je	.Lfound_v18_16
	vl	%v16,32(%r5,%r3)
	vst	%v18,16(%r5,%r2)
	vfenezbs %v17,%v16,%v16
	je	.Lfound_v16_32
	vl	%v18,48(%r5,%r3)
	vst	%v16,32(%r5,%r2)
	vfenezbs %v17,%v18,%v18
	je	.Lfound_v18_48
	vst	%v18,48(%r5,%r2)

	aghi	%r5,64
	lgr	%r1,%r5		/* If %r5 + 64 < maxlen? -> loop64.  */
	aghi	%r1,64
	clgrjl	%r1,%r4,.Lloop64

	vl	%v16,0(%r5,%r3)	/* Load s.  */
	j	.Llt64
END(STRNCPY_Z13)

# if ! HAVE_STRNCPY_IFUNC
strong_alias (STRNCPY_Z13, strncpy)
# endif

# if ! HAVE_STRNCPY_Z900_G5 && defined SHARED && IS_IN (libc)
strong_alias (STRNCPY_Z13, __GI_strncpy)
# endif
#endif
