/* Vector optimized 32/64 bit S/390 version of strcat.
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

#include <ifunc-strcat.h>
#if HAVE_STRCAT_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* char * strcat (const char *dest, const char *src)
   Concatenate two strings.

   Register usage:
   -r0=saved dest pointer for return
   -r1=tmp
   -r2=dest
   -r3=src
   -r4=tmp
   -r5=current_len
   -v16=part of src
   -v17=index of zero
   -v18=part of src
*/
ENTRY(STRCAT_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

	lgr	%r0,%r2		/* Save destination pointer for return.  */

	/* STRLEN
	   r1 = loaded bytes (tmp)
	   r4 = zero byte index (tmp)
	   r2 = dst
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
	vlgvb	%r4,%v16,7	/* Load byte index of zero.  */
	algr	%r5,%r4

.Llen_end:
	/* STRCPY
	   %r1 = loaded bytes (tmp)
	   %r4 = zero byte index (tmp)
	   %r3 = curr src pointer
	   %r2 = curr dst pointer
	 */
	la	%r2,0(%r5,%r2)	/* strcpy at end of dst-string.   */

	vlbb	%v16,0(%r3),6	/* Load s until next 4k-byte boundary.  */
	lcbb	%r1,0(%r3),6	/* Get bytes to 4k-byte boundary or 16.  */

	vfenezb	%v17,%v16,%v16	/* Find element not equal with zero search.  */
	vlgvb	%r5,%v17,7	/* Load zero index or 16 if not found.  */
	clrjl	%r5,%r1,.Lcpy_found_align /* If found zero within loaded bytes,
					     copy bytes before and return.  */

	/* Align s to 16 byte.  */
	risbgn	%r4,%r3,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r5,15		/* current_len = 15.  */
	slr	%r5,%r4		/* Compute highest index to 16byte boundary.  */

	vstl	%v16,%r5,0(%r2)	/* Copy loaded characters - no zero.  */
	ahi	%r5,1		/* Start loop at next character.  */

	/* Find zero in 16byte aligned loop.  */
.Lcpy_loop:
	vl	%v16,0(%r5,%r3)	/* Load s.  */
	vfenezbs %v17,%v16,%v16	/* Find element not equal with zero search.  */
	je	.Lcpy_found_v16_0 /* Jump away if zero was found.  */
	vl	%v18,16(%r5,%r3)/* Load next part of s.  */
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
	j	.Lcpy_loop	/* No zero -> loop.  */

.Lcpy_found_v16_32:
	aghi	%r5,32
.Lcpy_found_v16_0:
	la	%r4,0(%r5,%r2)
	vlgvb	%r1,%v17,7	/* Load byte index of zero.  */
	vstl	%v16,%r1,0(%r4)	/* Copy characters including zero.  */
	lgr	%r2,%r0		/* Load saved dest-ptr.  */
	br	%r14

.Lcpy_found_v18_48:
	aghi	%r5,32
.Lcpy_found_v18_16:
	la	%r4,16(%r5,%r2)
	vlgvb	%r1,%v17,7	/* Load byte index of zero.  */
	vstl	%v18,%r1,0(%r4)	/* Copy characters including zero.  */
	lgr	%r2,%r0		/* Load saved dest-ptr.  */
	br	%r14

.Lcpy_found_align:
	vstl	%v16,%r5,0(%r2)	/* Copy characters including zero.  */
	lgr	%r2,%r0		/* Load saved dest-ptr.  */
	br	%r14
END(STRCAT_Z13)

# if ! HAVE_STRCAT_IFUNC
strong_alias (STRCAT_Z13, strcat)
# endif

# if ! HAVE_STRCAT_C && defined SHARED && IS_IN (libc)
strong_alias (STRCAT_Z13, __GI_strcat)
# endif
#endif
