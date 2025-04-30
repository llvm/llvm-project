/* Vector Optimized 32/64 bit S/390 version of wmemset.
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

#include <ifunc-wmemset.h>
#if HAVE_WMEMSET_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* wchar_t *wmemset(wchar_t *dest, wchar_t wc, size_t n)
   Fill an array of wide-characters with a constant wide character
   and returns dest.

   Register usage:
   -r0=tmp
   -r1=tmp
   -r2=dest or current-pointer
   -r3=wc
   -r4=n
   -r5=tmp
   -v16=replicated wc
   -v17,v18,v19=copy of v16 for vstm
   -v31=saved dest for return
*/
ENTRY(WMEMSET_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

# if !defined __s390x__
	llgfr	%r4,%r4
# endif /* !defined __s390x__ */

	vlvgg	%v31,%r2,0	/* Save destination pointer for return.  */
	clgije	%r4,0,.Lend

	vlvgf	%v16,%r3,0	/* Generate vector with wchar_t wc.  */
	vrepf	%v16,%v16,0

	/* Check range of maxlen and convert to byte-count.  */
# ifdef __s390x__
	tmhh	%r4,49152	/* Test bit 0 or 1 of maxlen.  */
	lghi	%r5,-4		/* Max byte-count is 18446744073709551612.  */
# else
	tmlh	%r4,49152	/* Test bit 0 or 1 of maxlen.  */
	llilf	%r5,4294967292	/* Max byte-count is 4294967292.  */
# endif /* !__s390x__ */
	sllg	%r4,%r4,2	/* Convert character-count to byte-count.  */
	locgrne	%r4,%r5		/* Use max byte-count, if bit 0/1 was one.  */

	/* Align dest to 16 byte.  */
	risbg	%r0,%r2,60,128+63,0 /* Test if s is aligned and
				       %r3 = bits 60-63 'and' 15.  */
	je	.Lpreloop	/* If s is aligned, loop aligned.  */
	tmll	%r2,3		/* Test if s is 4-byte aligned?  */
	jne	.Lfallback	/* And use common-code variant if not.  */
	lghi	%r1,16
	slr	%r1,%r0		/* Compute byte count to load (16-x).  */
	clgr	%r1,%r4
	locgrh	%r1,%r4		/* min (byte count, n)  */
	aghik	%r5,%r1,-1	/* vstl needs highest index.  */
	vstl	%v16,%r5,0(%r2)	/* Store remaining bytes.  */
	clgrje	%r1,%r4,.Lend	/* Return if n bytes where set.  */
	slgr	%r4,%r1		/* Compute remaining byte count.  */
	la	%r2,0(%r1,%r2)

.Lpreloop:
	/* Now we are 16-byte aligned.  */
	clgijl	%r4,17,.Lremaining
	srlg	%r1,%r4,8	/* Split into 256byte blocks */
	clgije	%r1,0,.Lpreloop64
	vlr	%v17,%v16
	vlr	%v18,%v16
	vlr	%v19,%v16

.Lloop256:
	vstm	%v16,%v19,0(%r2)
	vstm	%v16,%v19,64(%r2)
	vstm	%v16,%v19,128(%r2)
	vstm	%v16,%v19,192(%r2)
	la	%r2,256(%r2)
	brctg	%r1,.Lloop256	/* Loop until all blocks are processed.  */

	llgfr	%r4,%r4
	nilf	%r4,255		/* Get remaining bytes */
	je	.Lend		/* Skip store remaining bytes if zero.  */

.Lpreloop64:
	clgijl	%r4,17,.Lremaining
	clgijl	%r4,33,.Lpreloop16
	srlg	%r1,%r4,5	/* Split into 32byte blocks */

.Lloop32:
	vst	%v16,0(%r2)
	vst	%v16,16(%r2)
	la	%r2,32(%r2)
	brctg	%r1,.Lloop32	/* Loop until all blocks are processed.  */

	llgfr	%r4,%r4
	nilf	%r4,31		/* Get remaining bytes */
	je	.Lend		/* Skip store remaining bytes if zero.  */

.Lpreloop16:
	clgijl	%r4,17,.Lremaining
	srlg	%r1,%r4,4	/* Split into 16byte blocks */

.Lloop16:
	vst	%v16,0(%r2)
	la	%r2,16(%r2)
	brctg	%r1,.Lloop16	/* Loop until all blocks are processed.  */

	llgfr	%r4,%r4
	nilf	%r4,15		/* Get remaining bytes */
	je	.Lend		/* Skip store remaining bytes if zero.  */

.Lremaining:
	aghi	%r4,-1		/* vstl needs highest index.  */
	vstl	%v16,%r4,0(%r2)

.Lend:
	vlgvg	%r2,%v31,0	/* Load saved dest for return value.  */
	br	%r14
.Lfallback:
	srlg	%r4,%r4,2	/* Convert byte-count to character-count.  */
	jg	WMEMSET_C
END(WMEMSET_Z13)

# if ! HAVE_WMEMSET_IFUNC
strong_alias (WMEMSET_Z13, __wmemset)
weak_alias (__wmemset, wmemset)
# endif

# if defined HAVE_S390_MIN_Z13_ZARCH_ASM_SUPPORT \
	&& defined SHARED && IS_IN (libc)
strong_alias (WMEMSET_Z13, __GI___wmemset)
weak_alias (WMEMSET_Z13, __GI_wmemset)
# endif
#endif
