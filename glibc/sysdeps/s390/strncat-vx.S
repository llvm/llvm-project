/* Vector optimized 32/64 bit S/390 version of strncat.
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

#include <ifunc-strncat.h>
#if HAVE_STRNCAT_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* char * strncat (const char *dest, const char *src, size_t n)
   Concatenate two strings - at most n characters of src.

   Register usage:
   -r0=saved dest pointer for return
   -r1=tmp
   -r2=dest
   -r3=src
   -r4=n
   -r5=current_len
   -r6=tmp
   -r7=tmp
   -v16=part of src
   -v17=index of zero
   -v18=part of src
   -v31=register save area for r6, r7
*/
ENTRY(STRNCAT_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

# if !defined __s390x__
	llgfr	%r4,%r4
# endif /* !defined __s390x__ */

	clgfi	%r4,0
	ber	%r14		/* Nothing to do, if n == 0.  */
	lgr	%r0,%r2		/* Save destination pointer for return.  */
	vlvgp	%v31,%r6,%r7	/* Save registers.  */

	/* STRLEN
	   %r1 = loaded bytes (tmp)
	   %r6 = zero byte index (tmp)
	   %r2 = dst
	*/
	vlbb	%v16,0(%r2),6	/* Load s until next 4k-byte boundary.  */
	lcbb	%r1,0(%r2),6	/* Get bytes to 4k-byte boundary or 16.  */

	vfenezb	%v16,%v16,%v16	/* Find element not equal with zero search.  */
	vlgvb	%r5,%v16,7	/* Load zero index or 16 if not found.  */
	clrjl	%r5,%r1,.Llen_end /* Found zero within loaded bytes, end.  */

	/* Align s to 16 byte.  */
	risbgn	%r1,%r2,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r5,16		/* current_len = 16.  */
	slr	%r5,%r1		/* Compute bytes to 16bytes boundary.  */

	/* Find zero in 16byte aligned loop.  */
.Llen_loop:
	vl	%v16,0(%r5,%r2)	/* Load s.  */
	vfenezbs %v16,%v16,%v16	/* Find element not equal with zero search.  */
	je	.Llen_found	/* Jump away if zero was found.  */
	vl	%v16,16(%r5,%r2)
	vfenezbs %v16,%v16,%v16
	je	.Llen_found16
	vl	%v16,32(%r5,%r2)
	vfenezbs %v16,%v16,%v16
	je	.Llen_found32
	vl	%v16,48(%r5,%r2)
	vfenezbs %v16,%v16,%v16
	je	.Llen_found48

	aghi	%r5,64
	j	.Llen_loop	/* No zero -> loop.  */

.Llen_found48:
	aghi	%r5,16
.Llen_found32:
	aghi	%r5,16
.Llen_found16:
	aghi	%r5,16
.Llen_found:
	vlgvb	%r1,%v16,7	/* Load byte index of zero.  */
	algr	%r5,%r1

.Llen_end:
	/* STRCPY
	   %r1 = zero byte index (tmp)
	   %r6 = loaded bytes (tmp)
	   %r3 = curr src pointer
	   %r2 = curr dst pointer
	   %r7 = border, tmp
	*/
	la	%r2,0(%r5,%r2)	/* strcpy at end of dst-string.  */

	vlbb	%v16,0(%r3),6	/* Load s until next 4k-byte boundary.  */
	lcbb	%r6,0(%r3),6	/* Get bytes to 4k-byte boundary or 16.  */
	llgfr	%r6,%r6		/* Convert 32bit to 64bit.  */

	lghi	%r5,0		/* current_len = 0.  */

	clgrjle	%r4,%r6,.Lcpy_remaining_v16 /* If n <= loaded-bytes
					       -> process remaining.  */

	/* n > loaded-byte-count.  */
	vfenezb	%v17,%v16,%v16	/* Find element not equal with zero search.  */
	vlgvb	%r1,%v17,7	/* Load zero index or 16 if not found.  */
	clrjl	%r1,%r6,.Lcpy_found_v16_store /* Found zero within loaded
						 bytes, copy and return.  */

	/* Align s to 16 byte.  */
	risbgn	%r7,%r3,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r5,15		/* current_len = 15.  */
	slr	%r5,%r7		/* Compute highest index to 16byte boundary.  */

	/* Zero not found and n > loaded-byte-count.  */
	vstl	%v16,%r5,0(%r2)	/* Copy loaded characters - no zero.  */
	ahi	%r5,1		/* Start loop at next character.  */

	/*
	  Now we are 16byte aligned, so we can load a full vreg
	  without page fault.
	 */
	lgr	%r1,%r5		/* If %r5 + 64 < maxlen? -> loop64.  */
	aghi	%r1,64
	clgrjl	%r1,%r4,.Lcpy_loop64

	vl	%v16,0(%r5,%r3)	/* Load s.  */
	clgijl	%r4,17,.Lcpy_remaining_v16 /* If n <=16,
					       process remaining bytes.  */
.Lcpy_lt64:
	lgr	%r7,%r4
	slgfi	%r7,16		/* border_len = n - 16.  */

	/* If current_len >= border then process remaining bytes.  */
	clgrjhe	%r5,%r7,.Lcpy_remaining_v16
	vfenezbs %v17,%v16,%v16	/* Find element not equal with zero search.  */
	je	.Lcpy_found_v16 /* Jump away if zero was found.  */
	vl	%v18,16(%r5,%r3) /* Load next part of s.  */
	vst	%v16,0(%r5,%r2)	/* Store previous part without zero to dst.  */
	aghi	%r5,16

	clgrjhe	%r5,%r7,.Lcpy_remaining_v18
	vfenezbs %v17,%v18,%v18
	je	.Lcpy_found_v18
	vl	%v16,16(%r5,%r3)
	vst	%v18,0(%r5,%r2)
	aghi	%r5,16

	clgrjhe	%r5,%r7,.Lcpy_remaining_v16
	vfenezbs %v17,%v16,%v16
	je	.Lcpy_found_v16
	vl	%v18,16(%r5,%r3)
	vst	%v16,0(%r5,%r2)
	aghi	%r5,16

.Lcpy_remaining_v18:
	vlr	%v16,%v18
.Lcpy_remaining_v16:
	/* v16 contains the remaining bytes [1...16].
	   Store remaining bytes and append string-termination.  */
	vfenezb	%v17,%v16,%v16	/* Find element not equal with zero search.  */
	slgrk	%r7,%r4,%r5	/* Remaining bytes = maxlen - current_len.  */
	aghi	%r7,-1		/* vstl needs highest index.  */
	vlgvb	%r1,%v17,7	/* Load zero index or 16 if not found.  */
	la	%r2,0(%r5,%r2)	/* vstl has no index register.  */
	/* Zero-index within remaining-bytes, store up to zero and end.  */
	clgrjle	%r1,%r7,.Lcpy_found_v16_store
	vstl	%v16,%r7,0(%r2)	/* Store remaining bytes.  */
	lghi	%r1,0
	stc	%r1,1(%r7,%r2)	/* Store string-null-termination beyond n.  */
.Lcpy_end:
	/* Restore saved registers.  */
	vlgvg	%r6,%v31,0
	vlgvg	%r7,%v31,1
	lgr	%r2,%r0		/* Load saved dest-ptr.  */
	br	%r14

.Lcpy_found_v16_32:
	aghi	%r5,32
	j	.Lcpy_found_v16
.Lcpy_found_v18_48:
	aghi	%r5,32
.Lcpy_found_v18_16:
	aghi	%r5,16
.Lcpy_found_v18:
	vlr	%v16,%v18
.Lcpy_found_v16:
	/* v16 contains a zero. Store remaining bytes to zero. current_len
	   has not reached border, thus checking for n is not needed!  */
	vlgvb	%r1,%v17,7	/* Load byte index of zero.  */
	la	%r2,0(%r5,%r2)
.Lcpy_found_v16_store:
	vstl	%v16,%r1,0(%r2)	/* Copy characters including zero.  */
	j	.Lcpy_end

	/* Find zero in 16byte aligned loop.  */
.Lcpy_loop64:
	vl	%v16,0(%r5,%r3)	/* Load s.  */
	vfenezbs %v17,%v16,%v16	/* Find element not equal with zero search.  */
	je	.Lcpy_found_v16 /* Jump away if zero was found.  */
	vl	%v18,16(%r5,%r3) /* Load next part of s.  */
	vst	%v16,0(%r5,%r2)	/* Store previous part without zero to dst.  */
	vfenezbs %v17,%v18,%v18
	je	.Lcpy_found_v18_16
	vl	%v16,32(%r5,%r3)
	vst	%v18,16(%r5,%r2)
	vfenezbs %v17,%v16,%v16
	je	.Lcpy_found_v16_32
	vl	%v18,48(%r5,%r3)
	vst	%v16,32(%r5,%r2)
	vfenezbs %v17,%v18,%v18
	je	.Lcpy_found_v18_48
	vst	%v18,48(%r5,%r2)

	aghi	%r5,64
	lgr	%r1,%r5		/* If %r5 + 64 < maxlen? -> loop64.  */
	aghi	%r1,64
	clgrjl	%r1,%r4,.Lcpy_loop64

	vl	%v16,0(%r5,%r3)	/* Load s.  */
	j	.Lcpy_lt64
END(STRNCAT_Z13)

# if ! HAVE_STRNCAT_IFUNC
strong_alias (STRNCAT_Z13, strncat)
# endif

# if ! HAVE_STRNCAT_C
/* See string/strncat.c and define STRNCAT_PRIMARY.  */
strong_alias (STRNCAT_Z13, __strncat)
#  if defined SHARED && IS_IN (libc)
strong_alias (__strncat, __GI___strncat)
#  endif
# endif
#endif
