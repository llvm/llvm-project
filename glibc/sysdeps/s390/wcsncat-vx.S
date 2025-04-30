/* Vector optimized 32/64 bit S/390 version of wcsncat.
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

#include <ifunc-wcsncat.h>
#if HAVE_WCSNCAT_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* wchar_t * wcsncat (wchar_t *dest, const wchar_t *src, size_t n)
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
ENTRY(WCSNCAT_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

# if !defined __s390x__
	llgfr	%r4,%r4
# endif /* !defined __s390x__ */

	clgfi	%r4,0
	ber	%r14		/* Nothing to do, if n == 0.  */

	vlbb	%v16,0(%r2),6	/* Load s until next 4k-byte boundary.  */
	lcbb	%r1,0(%r2),6	/* Get bytes to 4k-byte boundary or 16.  */

	/* If either src or dest is not 4byte aligned, use __wcsncat_c.  */
	tmll	%r2,3		/* Test if s is 4-byte aligned?   */
	jne	.Lfallback	/* And use common-code variant if not.  */
	tmll	%r3,3		/* Test if src is 4-byte aligned?   */
	jne	.Lfallback	/* And use common-code variant if not.  */

	lgr	%r0,%r2		/* Save destination pointer for return.  */
	vlvgp	%v31,%r6,%r7	/* Save registers.  */

	/* WCSLEN
	   %r1 = loaded bytes (tmp)
	   %r6 = zero byte index (tmp)
	   %r2 = dst
	*/
	vfenezf	%v16,%v16,%v16	/* Find element not equal with zero search.  */
	vlgvb	%r5,%v16,7	/* Load zero index or 16 if not found.  */
	clrjl	%r5,%r1,.Llen_end /* Found zero within loaded bytes, end.  */

	/* Align s to 16 byte.  */
	risbgn	%r1,%r2,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r5,16		/* current_len = 16.  */
	slr	%r5,%r1		/* Compute bytes to 16bytes boundary.  */

	/* Find zero in 16byte aligned loop.  */
.Llen_loop:
	vl	%v16,0(%r5,%r2)	/* Load s.  */
	vfenezfs %v16,%v16,%v16	/* Find element not equal with zero search.  */
	je	.Llen_found	/* Jump away if zero was found.  */
	vl	%v16,16(%r5,%r2)
	vfenezfs %v16,%v16,%v16
	je	.Llen_found16
	vl	%v16,32(%r5,%r2)
	vfenezfs %v16,%v16,%v16
	je	.Llen_found32
	vl	%v16,48(%r5,%r2)
	vfenezfs %v16,%v16,%v16
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
	/* WCSNCPY
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

	clgrjle	%r4,%r6,.Lcpy_remaining_v16 /* If n <= loaded-bytes
					       -> process remaining.  */

	/* n > loaded-byte-count.  */
	vfenezf	%v17,%v16,%v16	/* Find element not equal with zero search.  */
	vlgvb	%r1,%v17,7	/* Load zero index or 16 if not found.  */
	clrjl	%r1,%r6,.Lcpy_found_v16_store /* Found zero within loaded bytes,
					         copy and return.  */

	/* Align s to 16 byte.  */
	risbgn	%r1,%r3,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r5,15		/* current_len = 15.  */
	slr	%r5,%r1		/* Compute highest index to 16byte boundary.  *

	/* Zero not found and maxlen > loaded-byte-count.  */
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

	clgrjhe	%r5,%r7,.Lcpy_remaining_v16
	vfenezfs %v17,%v16,%v16	/* Find element not equal with zero search.  */
	je	.Lcpy_found_v16 /* Jump away if zero was found.  */
	vl	%v18,16(%r5,%r3) /* Load next part of s.  */
	vst	%v16,0(%r5,%r2)	/* Save previous part without zero to dst.  */
	aghi	%r5,16

	clgrjhe	%r5,%r7,.Lcpy_remaining_v18
	vfenezfs %v17,%v18,%v18
	je	.Lcpy_found_v18
	vl	%v16,16(%r5,%r3)
	vst	%v18,0(%r5,%r2)
	aghi	%r5,16

	clgrjhe	%r5,%r7,.Lcpy_remaining_v16
	vfenezfs %v17,%v16,%v16
	je	.Lcpy_found_v16
	vl	%v18,16(%r5,%r3)
	vst	%v16,0(%r5,%r2)
	aghi	%r5,16

.Lcpy_remaining_v18:
	vlr	%v16,%v18
.Lcpy_remaining_v16:
	/* v16 contains the remaining bytes [1...16].
	   Store remaining bytes and append string-termination.  */
	vfenezf	%v17,%v16,%v16	/* Find element not equal with zero search.  */
	slgrk	%r7,%r4,%r5	/* Remaining bytes = maxlen - current_len.  */
	aghi	%r7,-1		/* vstl needs highest index.  */
	vlgvb	%r1,%v17,7	/* Load zero index or 16 if not found.  */
	la	%r2,0(%r5,%r2)	/* vstl has no index register.  */
	/* Zero-index within remaining-bytes, store up to zero and end.  */
	clgrjle	%r1,%r7,.Lcpy_found_v16_store
	vstl	%v16,%r7,0(%r2)	/* Store remaining bytes.  */
	lghi	%r1,0
	st	%r1,1(%r7,%r2)	/* Store string-null-termination beyond n.  */
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
	aghi	%r1,3		/* Also copy remaining bytes of zero.  */
	vstl	%v16,%r1,0(%r2)	/* Copy characters including zero.  */
	j	.Lcpy_end

	/* Find zero in 16byte aligned loop.  */
.Lcpy_loop2:
	vl	%v16,16(%r5,%r3)
	vst	%v18,0(%r5,%r2)
	aghi	%r5,16

.Lcpy_loop64:
	vl	%v16,0(%r5,%r3)
	vfenezfs %v17,%v16,%v16	/* Find element not equal with zero search.  */
	je	.Lcpy_found_v16 /* Jump away if zero was found.  */
	vl	%v18,16(%r5,%r3) /* Load next part of s.  */
	vst	%v16,0(%r5,%r2)	/* Save previous part without zero to dst.  */
	vfenezfs %v17,%v18,%v18
	je	.Lcpy_found_v18_16
	vl	%v16,32(%r5,%r3)
	vst	%v18,16(%r5,%r2)
	vfenezfs %v17,%v16,%v16
	je	.Lcpy_found_v16_32
	vl	%v18,48(%r5,%r3)
	vst	%v16,32(%r5,%r2)
	vfenezfs %v17,%v18,%v18
	je	.Lcpy_found_v18_48
	vst	%v18,48(%r5,%r2)

	aghi	%r5,64
	lgr	%r1,%r5		/* If %r5 + 64 < maxlen? -> loop64.  */
	aghi	%r1,64
	clgrjl	%r1,%r4,.Lcpy_loop64

	vl	%v16,0(%r5,%r3)	/* Load s.  */
	j	.Lcpy_lt64

.Lfallback:
	jg	WCSNCAT_C
END(WCSNCAT_Z13)

# if ! HAVE_WCSNCAT_IFUNC
strong_alias (WCSNCAT_Z13, wcsncat)
# endif
#endif
