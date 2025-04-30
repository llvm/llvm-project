/* Vector optimized 32/64 bit S/390 version of wcsnlen.
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

#include <ifunc-wcsnlen.h>
#if HAVE_WCSNLEN_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* size_t wcsnlen (const wchar_t *s, size_t maxlen)
   Returns the number of characters in s or at most maxlen.

   Register usage:
   -r1=tmp
   -r2=address of string
   -r3=maxlen (number of characters to be read)
   -r4=tmp
   -r5=current_len and return_value
   -v16=part of s
*/
ENTRY(WCSNLEN_Z13)

	.machine "z13"
	.machinemode "zarch_nohighgprs"

# if !defined __s390x__
	llgfr	%r3,%r3
# endif /* !defined __s390x__ */

	clgfi	%r3,0		/* if maxlen == 0, return 0.  */
	locgre	%r2,%r3
	ber	%r14

	vlbb	%v16,0(%r2),6	/* Load s until next 4k-byte boundary.  */
	lcbb	%r1,0(%r2),6	/* Get bytes to 4k-byte boundary or 16.  */
	llgfr	%r1,%r1		/* Convert 32bit to 64bit.  */

	tmll	%r2,3		/* Test if s is 4-byte aligned?  */
	jne	.Lfallback	/* And use common-code variant if not.  */

	/* Check range of maxlen and convert to byte-count.  */
# ifdef __s390x__
	tmhh	%r3,49152	/* Test bit 0 or 1 of maxlen.  */
	lghi	%r4,-4		/* Max byte-count is 18446744073709551612.  */
# else
	tmlh	%r3,49152	/* Test bit 0 or 1 of maxlen.  */
	llilf	%r4,4294967292	/* Max byte-count is 4294967292.  */
# endif /* !__s390x__ */
	sllg	%r3,%r3,2	/* Convert character-count to byte-count.  */
	locgrne	%r3,%r4		/* Use max byte-count, if bit 0/1 was one.  */

	vfenezf	%v16,%v16,%v16	/* Find element not equal with zero search.  */
	clgr	%r1,%r3
	locgrh	%r1,%r3		/* loaded_byte_count
				   = min (loaded_byte_count, maxlen)  */

	vlgvb	%r5,%v16,7	/* Load zero index or 16 if not found.  */
	clrjl	%r5,%r1,.Lend	/* Found zero within loaded bytes -> return.  */

	clgr	%r1,%r3		/* If loaded_byte_count == maxlen -> end.  */
	locgre	%r5,%r3
	je	.Lend

	/* Align s to 16 byte.  */
	risbgn	%r4,%r2,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r5,16		/* current_len = 16.  */
	slr	%r5,%r4		/* Compute bytes to 16bytes boundary.  */

	lgr	%r1,%r5		/* If %r5 + 64 < maxlen? -> loop64.  */
	aghi	%r1,64
	clgrjl	%r1,%r3,.Lloop64

	/* Find zero in max 64byte with aligned s.  */
.Llt64:
	vl	%v16,0(%r5,%r2)	/* Load s.  */
	vfenezfs %v16,%v16,%v16	/* Find element not equal with zero search.  */
	je	.Lfound		/* Jump away if zero was found.  */
	aghi	%r5,16
	clgrjhe	%r5,%r3,.Lfound /* If current_len >= maxlen -> end.  */
	vl	%v16,0(%r5,%r2)
	vfenezfs %v16,%v16,%v16
	je	.Lfound
	aghi	%r5,16
	clgrjhe	%r5,%r3,.Lfound
	vl	%v16,0(%r5,%r2)
	vfenezfs %v16,%v16,%v16
	je	.Lfound
	aghi	%r5,16
	clgrjhe	%r5,%r3,.Lfound
	vl	%v16,0(%r5,%r2)
	vfenezfs %v16,%v16,%v16
	j	.Lfound

.Lfound48:
	aghi	%r5,16
.Lfound32:
	aghi	%r5,16
.Lfound16:
	aghi	%r5,16
.Lfound:
	vlgvb	%r4,%v16,7	/* Load byte index of zero or 16 if no zero.  */
	algr	%r5,%r4

	clgr	%r5,%r3
	locgrh	%r5,%r3		/* Return min (current_len, maxlen).  */
.Lend:
	srlg	%r2,%r5,2	/* Convert byte-count to character-count.  */
	br	%r14

	/* Find zero in 16byte aligned loop.  */
.Lloop64:
	vl	%v16,0(%r5,%r2)	/* Load s.  */
	vfenezfs %v16,%v16,%v16	/* Find element not equal with zero search.  */
	je	.Lfound		/* Jump away if zero was found.  */
	vl	%v16,16(%r5,%r2)
	vfenezfs %v16,%v16,%v16
	je	.Lfound16
	vl	%v16,32(%r5,%r2)
	vfenezfs %v16,%v16,%v16
	je	.Lfound32
	vl	%v16,48(%r5,%r2)
	vfenezfs %v16,%v16,%v16
	je	.Lfound48

	aghi	%r5,64
	lgr	%r1,%r5		/* If %r5 + 64 < maxlen? -> loop64.  */
	aghi	%r1,64
	clgrjl	%r1,%r3,.Lloop64

	j	.Llt64

.Lfallback:
	jg	WCSNLEN_C
END(WCSNLEN_Z13)

# if ! HAVE_WCSNLEN_IFUNC
strong_alias (WCSNLEN_Z13, __wcsnlen)
weak_alias (__wcsnlen, wcsnlen)
# endif
#endif
