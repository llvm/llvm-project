/* Vector optimized 32/64 bit S/390 version of strnlen.
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

#include <ifunc-strnlen.h>

#if HAVE_STRNLEN_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* size_t strnlen (const char *s, size_t maxlen)
   Returns the number of characters in s or at most maxlen.

   Register usage:
   -r1=tmp
   -r2=address of string
   -r3=maxlen (number of characters to be read)
   -r4=tmp
   -r5=current_len and return_value
   -v16=part of s
*/
ENTRY(STRNLEN_Z13)
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

	vfenezb	%v16,%v16,%v16	/* Find element not equal with zero search.  */
	clgr	%r1,%r3
	locgrh	%r1,%r3		/* loaded_byte_count
				   = min (loaded_byte_count, maxlen)  */

	vlgvb	%r5,%v16,7	/* Load zero index or 16 if not found.  */
	clr	%r5,%r1		/* If found zero within loaded bytes?  */
	locgrl	%r2,%r5		/* Then copy return value.  */
	blr	%r14		/* And return.  */

	clgr	%r1,%r3		/* If loaded_byte_count == maxlen?  */
	locgre	%r2,%r3		/* Then copy return value.  */
	ber	%r14		/* And return.  */

	/* Align s to 16 byte.  */
	risbgn	%r4,%r2,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r5,16		/* current_len = 16.  */
	slr	%r5,%r4		/* Compute bytes to 16bytes boundary.  */

	lgr	%r1,%r5		/* If %r5 + 64 < maxlen? -> loop64.  */
	aghi	%r1,64
	clgrjl	%r1,%r3,.Lloop64

	/* Find zero in max 64byte with aligned s.  */
.Llt64:
	vl	%v16,0(%r5,%r2) /* Load s.  */
	vfenezbs %v16,%v16,%v16	/* Find element not equal with zero search.  */
	je	.Lfound		/* Jump away if zero was found.  */
	aghi	%r5,16
	clgrjhe	%r5,%r3,.Lfound /* current_len >= maxlen -> end.  */
	vl	%v16,0(%r5,%r2)
	vfenezbs %v16,%v16,%v16
	je	.Lfound
	aghi	%r5,16
	clgrjhe	%r5,%r3,.Lfound
	vl	%v16,0(%r5,%r2)
	vfenezbs %v16,%v16,%v16
	je	.Lfound
	aghi	%r5,16
	clgrjhe	%r5,%r3,.Lfound
	vl	%v16,0(%r5,%r2)
	vfenezbs %v16,%v16,%v16
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
	lgr	%r2,%r5
	br	%r14

	/* Find zero in 16 byte aligned loop.  */
.Lloop64:
	vl	%v16,0(%r5,%r2) /* Load s.  */
	vfenezbs %v16,%v16,%v16	/* Find element not equal with zero search.  */
	je	.Lfound		/* Jump away if zero was found.  */
	vl	%v16,16(%r5,%r2)
	vfenezbs %v16,%v16,%v16
	je	.Lfound16
	vl	%v16,32(%r5,%r2)
	vfenezbs %v16,%v16,%v16
	je	.Lfound32
	vl	%v16,48(%r5,%r2)
	vfenezbs %v16,%v16,%v16
	je	.Lfound48

	aghi	%r5,64
	lgr	%r1,%r5		/* If %r5 + 64 < maxlen? -> loop64.  */
	aghi	%r1,64
	clgrjl	%r1,%r3,.Lloop64

	j	.Llt64
END(STRNLEN_Z13)

# if ! HAVE_STRNLEN_IFUNC
strong_alias (STRNLEN_Z13, __strnlen)
weak_alias (__strnlen, strnlen)
# endif

# if ! HAVE_STRNLEN_C && defined SHARED && IS_IN (libc)
strong_alias (STRNLEN_Z13, __GI_strnlen)
strong_alias (STRNLEN_Z13, __GI___strnlen)
# endif

#endif /* HAVE_STRNLEN_Z13  */
