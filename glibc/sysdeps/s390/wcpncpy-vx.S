/* Vector optimized 32/64 bit S/390 version of wcpncpy.
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

#include <ifunc-wcpncpy.h>
#if HAVE_WCPNCPY_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* wchar_t * wcpncpy (wchar_t *dest, const wchar_t *src, size_t n)
   Copies at most n characters of string src to dest
   returning a pointer to its end or dest+n
   if src is smaller than n.

   Register usage:
   -%r0 = return value
   -%r1 = zero byte index
   -%r2 = curr dst pointer
   -%r3 = curr src pointer
   -%r4 = n
   -%r5 = current_len
   -%r6 = loaded bytes
   -%r7 = border, tmp
*/
ENTRY(WCPNCPY_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

# if !defined __s390x__
	llgfr	%r4,%r4
# endif /* !defined __s390x__ */

	clgfi	%r4,0
	ber	%r14		/* Nothing to do, if n == 0.  */

	vlbb	%v16,0(%r3),6	/* Load s until next 4k-byte boundary.  */

	tmll	%r3,3		/* Test if s is 4-byte aligned?  */
	jne	.Lfallback	/* And use common-code variant if not.  */

	vlvgp	%v31,%r6,%r7	/* Save registers.  */
	lghi	%r5,0		/* current_len = 0.  */

	lcbb	%r6,0(%r3),6	/* Get bytes to 4k-byte boundary or 16.  */
	llgfr	%r6,%r6		/* Convert 32bit to 64bit.  */

	/* Check range of maxlen and convert to byte-count.  */
# ifdef __s390x__
	tmhh	%r4,49152	/* Test bit 0 or 1 of maxlen.  */
	lghi	%r1,-4		/* Max byte-count is 18446744073709551612.  */
# else
	tmlh	%r4,49152	/* Test bit 0 or 1 of maxlen.  */
	llilf	%r1,4294967292	/* Max byte-count is 4294967292.  */
# endif /* !__s390x__ */
	sllg	%r4,%r4,2	/* Convert character-count to byte-count.  */
	locgrne	%r4,%r1		/* Use max byte-count, if bit 0/1 was one.  */

	la	%r0,0(%r4,%r2)	/* Save destination pointer + n for return.  */

	clgrjle	%r4,%r6,.Lremaining_v16 /* If n <= loaded-bytes
					   -> process remaining.  */

	/* n > loaded-byte-count */
	vfenezf	%v17,%v16,%v16	/* Find element not equal with zero search.  */
	vlgvb	%r1,%v17,7	/* Load zero index or 16 if not found.  */
	aghi	%r1,3		/* Also copy remaining bytes of zero.  */
	clrjl	%r1,%r6,.Lfound_v16_store /* Found zero within loaded bytes,
					     copy and return.  */

	/* Align s to 16 byte.  */
	risbgn	%r7,%r3,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r5,15		/* current_len = 15.  */
	slr	%r5,%r7		/* Compute highest index to 16byte boundary.  */

	/* Zero not found and n > loaded-byte-count.  */
	vstl	%v16,%r5,0(%r2)	/* Copy loaded characters - no zero.  */
	ahi	%r5,1		/* Start loop at next character.  */

	/* Now we are 16byte aligned, so we can load a full vreg
	   without page fault.  */
	lgr	%r1,%r5		/* If %r5 + 64 < maxlen? -> loop64.  */
	aghi	%r1,64
	clgrjl	%r1,%r4,.Lloop64

	vl	%v16,0(%r5,%r3)	/* Load s.  */
	clgijl	%r4,17,.Lremaining_v16	/* If n <=16,
					   process remaining bytes.  */
.Llt64:
	lgr	%r7,%r4
	slgfi	%r7,16		/* border_len = n - 16.  */

	clgrjhe	%r5,%r7,.Lremaining_v16 /* If current_len >= border
					   then process remaining bytes.  */
	vfenezfs %v17,%v16,%v16	/* Find element not equal with zero search.  */
	je	.Lfound_v16	/* Jump away if zero was found.  */
	vl	%v18,16(%r5,%r3) /* Load next part of s.  */
	vst	%v16,0(%r5,%r2)	/* Store previous part without zero to dst.  */
	aghi	%r5,16

	clgrjhe	%r5,%r7,.Lremaining_v18
	vfenezfs %v17,%v18,%v18
	je	.Lfound_v18
	vl	%v16,16(%r5,%r3)
	vst	%v18,0(%r5,%r2)
	aghi	%r5,16

	clgrjhe	%r5,%r7,.Lremaining_v16
	vfenezfs %v17,%v16,%v16
	je	.Lfound_v16
	vl	%v18,16(%r5,%r3)
	vst	%v16,0(%r5,%r2)
	aghi	%r5,16

.Lremaining_v18:
	vlr	%v16,%v18
.Lremaining_v16:
	/* v16 contains the remaining bytes [1...16].
	   Store remaining bytes and append string-termination.  */
	vfenezf	%v17,%v16,%v16	/* Find element not equal with zero search.  */
	slgrk	%r7,%r4,%r5	/* Remaining bytes = maxlen - current_len  */
	aghi	%r7,-1		/* vstl needs highest index.  */
	la	%r2,0(%r5,%r2)	/* vstl has no index register.  */
	vlgvb	%r1,%v17,7	/* Load zero index or 16 if not found.  */
	aghi	%r1,3		/* Also copy remaining bytes of zero.  */
	/* Zero in remaining bytes? -> jump away (zero-index <= max-index).  */
	clrjle	%r1,%r7,.Lfound_v16_store
	vstl	%v16,%r7,0(%r2)	/* Store remaining bytes without null
				   termination!  */
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
	   has not reached border, thus checking for n is not needed!  */
	vlgvb	%r1,%v17,7	/* Load byte index of zero.  */
	la	%r2,0(%r5,%r2)	/* vstl has no support for index-register.  */
	aghi	%r1,3		/* Also copy remaining bytes of zero.  */
.Lfound_v16_store:
	vstl	%v16,%r1,0(%r2)	/* Copy characters including zero.  */
	/* Fill remaining bytes with zero - remaining byte count always > 0.  */
	algr	%r5,%r1		/* Remaining bytes (=%r4) = ...  */
	slgr	%r4,%r5		/* = n - (currlen + zero_index + 1) */
	la	%r2,0(%r1,%r2)	/* Pointer to zero. start filling beyond.  */
	lay	%r0,-3(%r2)	/* Save return-pointer to found zero.  */
	clgije	%r4,1,.Lend	/* Skip zero-filling, if found-zero is last
				   possible character.
				   (1 is substracted from r4 below!).  */
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

	/* Find zero in 16byte aligned loop.  */
.Lloop64:
	vl	%v16,0(%r5,%r3)
	vfenezfs %v17,%v16,%v16	/* Find element not equal with zero search.  */
	je	.Lfound_v16	/* Jump away if zero was found.  */
	vl	%v18,16(%r5,%r3) /* Load next part of s.  */
	vst	%v16,0(%r5,%r2)	/* Store previous part without zero to dst.  */
	vfenezfs %v17,%v18,%v18
	je	.Lfound_v18_16
	vl	%v16,32(%r5,%r3)
	vst	%v18,16(%r5,%r2)
	vfenezfs %v17,%v16,%v16
	je	.Lfound_v16_32
	vl	%v18,48(%r5,%r3)
	vst	%v16,32(%r5,%r2)
	vfenezfs %v17,%v18,%v18
	je	.Lfound_v18_48
	vst	%v18,48(%r5,%r2)

	aghi	%r5,64
	lgr	%r1,%r5		/* If %r5 + 64 < maxlen? -> loop64.  */
	aghi	%r1,64
	clgrjl	%r1,%r4,.Lloop64

	vl	%v16,0(%r5,%r3)	/* Load s.  */
	j	.Llt64

.Lfallback:
	jg	WCPNCPY_C
END(WCPNCPY_Z13)

# if ! HAVE_WCPNCPY_IFUNC
strong_alias (WCPNCPY_Z13, __wcpncpy)
weak_alias (__wcpncpy, wcpncpy)
# endif
#endif
