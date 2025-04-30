/* Vector optimized 32/64 bit S/390 version of strspn.
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

#include <ifunc-strspn.h>

#if HAVE_STRSPN_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* size_t strspn (const char *s, const char * accept)
   The strspn() function calculates the length of the initial segment
   of s which consists entirely of characters in accept.

   This method checks the length of accept string. If it fits entirely
   in one vector register, a fast algorithm is used, which does not need
   to check multiple parts of accept-string. Otherwise a slower full
   check of accept-string is used.

   register overview:
   r3:  pointer to start of accept-string
   r2:  pointer to start of search-string
   r4:  loaded byte count of vl search-string
   r0:  found byte index
   r1:  current return len of s
   v16: search-string
   v17: accept-string
   v18: temp-vreg

   ONLY FOR SLOW:
   v19: first accept-string
   v20: zero for preparing acc-vector
   v21: global mask; 1 indicates a match between
	search-string-vreg and any accept-character
   v22: current mask; 1 indicates a match between
	search-string-vreg and any accept-character in current acc-vreg
   v30, v31: for re-/storing registers r6, r8, r9
   r5:  current len of accept-string
   r6:	zero-index in search-string or 16 if no zero
	or min(zero-index, loaded byte count)
   r8:	>0, if former accept-string-part contains a zero,
	otherwise =0;
   r9: loaded byte count of vlbb accept-string
*/
ENTRY(STRSPN_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

	/*
	  Check if accept-string fits in one vreg:
	  ----------------------------------------
	*/
	vlbb	%v17,0(%r3),6	/* Load accept.  */
	lcbb	%r4,0(%r3),6
	jo	.Lcheck_onbb	/* Special case if accept lays
				   on block-boundary.  */
.Lcheck_notonbb:
	vistrbs	%v17,%v17	/* Fill with zeros after first zero.  */
	je	.Lfast		/* Zero found -> accept fits in one vreg.  */
	j	.Lslow		/* No zero -> accept exceeds one vreg.  */

.Lcheck_onbb:
	/* Accept lays on block-boundary.  */
	vfenezb	%v18,%v17,%v17	/* Search zero in loaded accept bytes.  */
	vlgvb	%r0,%v18,7	/* Get index of zero or 16 if not found.  */
	clrjl	%r0,%r4,.Lcheck_notonbb /* Zero index < loaded bytes count ->
					   Accept fits in one vreg;
					   Fill with zeros and proceed
					   with FAST.  */
	vl	%v17,0(%r3)	/* Load accept, which exceeds loaded bytes.  */
	j	.Lcheck_notonbb /* Check if accept fits in one vreg.  */


	/*
	  Search s for accept in one vreg
	  -------------------------------
	*/
.Lfast:
	/* Complete accept-string is in v17 and remaining bytes are zero.  */

	vlbb	%v16,0(%r2),6	/* Load s until next 4k-byte boundary.  */
	lcbb	%r1,0(%r2),6	/* Get bytes to 4k-byte boundary or 16.  */

	vfaezbs	%v16,%v16,%v17,8 /* Find first element in v16
				    unequal to any in v17
				    or first zero element.  */
	vlgvb	%r0,%v16,7	/* Load byte index of found element.  */
	/* If found index is within loaded bytes (%r0 < %r1),
	   return with found element index (=equal count).  */
	clr	%r0,%r1
	locgrl	%r2,%r0
	blr	%r14

	/* Align s to 16 byte.  */
	risbgn	%r4,%r2,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r1,16		/* current_len = 16.  */
	slr	%r1,%r4		/* Compute bytes to 16bytes boundary.  */

.Lfast_loop:
	vl	%v16,0(%r1,%r2)	/* Load search-string.  */
	vfaezbs	%v16,%v16,%v17,8 /* Find first element in v16
				    unequal to any in v17
				    or first zero element.  */
	jno	.Lfast_loop_found
	vl	%v16,16(%r1,%r2)
	vfaezbs	%v16,%v16,%v17,8
	jno	.Lfast_loop_found16
	vl	%v16,32(%r1,%r2)
	vfaezbs	%v16,%v16,%v17,8
	jno	.Lfast_loop_found32
	vl	%v16,48(%r1,%r2)
	vfaezbs	%v16,%v16,%v17,8
	jno	.Lfast_loop_found48

	aghi	%r1,64
	j	.Lfast_loop	/* Loop if no element was unequal to accept
				   and not zero.  */

	/* Found unequal or zero element.  */
.Lfast_loop_found48:
	aghi	%r1,16
.Lfast_loop_found32:
	aghi	%r1,16
.Lfast_loop_found16:
	aghi	%r1,16
.Lfast_loop_found:
	vlgvb	%r0,%v16,7	/* Load byte index of found element.  */
	algrk	%r2,%r1,%r0	/* And add it to current len.  */
	br	%r14


	/*
	  Search s for accept in multiple vregs
	  -------------------------------------
	*/
.Lslow:
	/* Save registers.  */
	vlvgg	%v30,%r6,0
	vlvgp	%v31,%r8,%r9
	lghi	%r1,0		/* current_len = 0.  */

	/* Accept in v17 without zero.  */
	vlr	%v19,%v17	/* Save first acc-part for a fast reload.  */
	vzero	%v20		/* Zero for preparing acc-vector.  */

	/* Align s to 16 byte.  */
	risbg	%r0,%r2,60,128+63,0 /* Test if s is aligned and
				     %r0 = bits 60-63 'and' 15  */
	je	.Lslow_loop_str /* If s is aligned, loop aligned */
	lghi	%r4,15
	slr	%r4,%r0		/* Compute highest index to load (15-x).  */
	vll	%v16,%r4,0(%r2) /* Load up to 16byte boundary (vll needs
				   highest index, left bytes are 0).  */
	ahi	%r4,1		/* Work with loaded byte count.  */
	vzero	%v21		/* Zero out global mask.  */
	lghi	%r5,0		/* Set current len of accept-string to zero.  */
	vfenezb	%v18,%v16,%v16	/* Find zero in current string-part.  */
	lghi	%r8,0		/* There is no zero in first accept-part.  */
	vlgvb	%r6,%v18,7	/* Load byte index of zero or 16
				   if there is no zero.  */
	clr	%r4,%r6		/* cc==1 if loaded byte count < zero-index.  */
	locrl	%r6,%r4		/* Load on cc==1.  */
	j	.Lslow_loop_acc

	/* Process s in 16byte aligned loop.  */
.Lslow_next_str:
	vlr	%v17,%v19	/* Load first part of accept (no zero).  */
	algfr	%r1,%r4		/* Add loaded byte count to current len.  */
.Lslow_loop_str:
	vl	%v16,0(%r1,%r2)	/* Load search-string.  */
	lghi	%r4,16		/* Loaded byte count is 16.  */
	vzero	%v21		/* Zero out global mask.  */
	lghi	%r5,0		/* Set current len of accept-string to zero.  */
	vfenezb	%v18,%v16,%v16	/* Find zero in current string-part.  */
	lghi	%r8,0		/* There is no zero in first accept-part.  */
	vlgvb	%r6,%v18,7	/* Load byte index of zero or 16 if no zero.  */

.Lslow_loop_acc:
	vfaeb	%v22,%v16,%v17,4 /* Create matching-mask (1 in mask ->
				    character matches any accepted character in
				    this accept-string-part) IN=0, RT=1.  */
	vo	%v21,%v21,%v22	/* global-mask = global- | matching-mask.  */
	vfenezb	%v18,%v21,%v21	/* Find first zero in global-mask.  */
	vlgvb	%r0,%v18,7	/* Get first found zero-index
				   (= first mismatch).  */
	clrjl	%r0,%r6,.Lslow_next_acc /* Mismatch-index < min(lbc,zero-index)
					    -> Process this string-part
					       with next acc-part.  */
	clrjhe	%r0,%r4,.Lslow_next_str /* Found-index >= loaded byte count
					   -> All loaded bytes are matching
					      any accept-character
					      and are not zero.  */
	/* All bytes are matching any characters in accept-string
	   and search-string is fully processed (found-index == zero-index)  */
.Lslow_add_lbc_end:
	algrk	%r2,%r1,%r0	/* Add matching characters to current_len.  */
	/* Restore registers.  */
	vlgvg	%r6,%v30,0
	vlgvg	%r8,%v31,0
	vlgvg	%r9,%v31,1
	br	%r14



.Lslow_next_acc:
	clijh	%r8,0,.Lslow_add_lbc_end /* There was a zero in last acc-part
					     -> Add found index to current len
						and end.  */
	vlbb	%v17,16(%r5,%r3),6 /* Load next accept part.  */
	aghi	%r5,16		/* Add current_len of accept-string.  */
	lcbb	%r9,0(%r5,%r3),6 /* Get loaded byte count of accept-string.  */
	jo	.Lslow_next_acc_onbb /* Jump away if accept-string is
					 on block-boundary.  */
.Lslow_next_acc_notonbb:
	vistrbs	%v17,%v17	/* Fill with zeros after first zero.  */
	jo	.Lslow_loop_acc /* No zero found -> no preparation needed.  */

.Lslow_next_acc_prepare_zero:
	/* Zero in accept-part: fill zeros with first-accept-character.  */
	vlgvb	%r8,%v17,0	/* Load first element of acc-part.  */
	clije	%r8,0,.Lslow_add_lbc_end /* End if zero is first character
					     in this part of accept-string.  */
	/* r8>0 -> zero found in this acc-part.  */
	vrepb	%v18,%v17,0	/* Replicate first char accross all chars.  */
	vceqb	%v22,%v20,%v17	/* Create a mask (v22) of null chars
				   by comparing with 0 (v20).  */
	vsel	%v17,%v18,%v17,%v22 /* Replace null chars with first char.  */
	j	.Lslow_loop_acc /* Accept part is prepared -> process.  */

.Lslow_next_acc_onbb:
	vfenezb	%v18,%v17,%v17	/* Find zero in loaded bytes of accept part.  */
	vlgvb	%r8,%v18,7	/* Load byte index of zero.  */
	clrjl	%r8,%r9,.Lslow_next_acc_notonbb /* Found a zero in loaded bytes
						    -> Prepare vr.  */
	vl	%v17,0(%r5,%r3)	/* Load over boundary ...  */
	lghi	%r8,0		/* r8=0 -> no zero in this part of acc,
				   Check for zero is in jump-target.  */
	j	.Lslow_next_acc_notonbb /* ... and search for zero in
					    fully loaded vreg again.  */
END(STRSPN_Z13)

# if ! HAVE_STRSPN_IFUNC
strong_alias (STRSPN_Z13, strspn)
# endif

# if ! HAVE_STRSPN_C && defined SHARED && IS_IN (libc)
strong_alias (STRSPN_Z13, __GI_strspn)
# endif

#endif /* HAVE_STRSPN_Z13  */
