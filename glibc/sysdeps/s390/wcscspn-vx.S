/* Vector optimized 32/64 bit S/390 version of wcscspn.
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

#include <ifunc-wcscspn.h>
#if HAVE_WCSCSPN_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* size_t wcscspn (const wchar_t *s, const wchar_t * reject)
   The wcscspn() function calculates the length of the initial segment
   of s which consists entirely of characters not in reject.

   This method checks the length of reject string. If it fits entirely
   in one vector register, a fast algorithm is used, which does not need
   to check multiple parts of accept-string. Otherwise a slower full
   check of accept-string is used.

   register overview:
   r3:  pointer to start of reject-string
   r2:  pointer to start of search-string
   r0:  loaded byte count of vlbb search-string
   r4:  found byte index
   r1:  current return len
   v16: search-string
   v17: reject-string
   v18: temp-vreg

   ONLY FOR SLOW:
   v19: first reject-string
   v20: zero for preparing acc-vector
   v21: global mask; 1 indicates a match between
	search-string-vreg and any reject-character
   v22: current mask; 1 indicates a match between
	search-string-vreg and any reject-character in current acc-vreg
   v30, v31: for re-/storing registers r6, r8, r9
   r5:  current len of reject-string
   r6:  zero-index in search-string or 16 if no zero
	or min(zero-index, loaded byte count)
   r8:  >0, if former reject-string-part contains a zero,
			otherwise =0;
   r9:  loaded byte count of vlbb reject-string
*/
ENTRY(WCSCSPN_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

	tmll	%r2,3		/* Test if s is 4-byte aligned?  */
	jne	.Lfallback	/* And use common-code variant if not.  */

	/*
	  Check if reject-string fits in one vreg:
	  ----------------------------------------
	*/
	vlbb	%v17,0(%r3),0	/* Load reject.  */
	lcbb	%r0,0(%r3),0
	jo	.Lcheck_onbb	/* Special case if reject
				   lays on block-boundary.  */

.Lcheck_notonbb:
	lghi	%r1,0		/* Zero out current len.  */
	vistrfs	%v17,%v17	/* Fill with zeros after first zero.  */
	je	.Lfast		/* Zero found -> reject fits in one vreg.  */
	j	.Lslow		/* No zero -> reject exceeds one vreg.  */


.Lcheck_onbb:
	/* Reject lays on block-boundary.  */
	nill	%r0,65532	/* Recognize only fully loaded characters.  */
	je	.Lcheck_onbb2	/* Reload vr, if we loaded no full wchar_t.  */
	vfenezf	%v18,%v17,%v17	/* Search zero in loaded reject bytes.  */
	vlgvb	%r4,%v18,7	/* Get index of zero or 16 if not found.  */
	clrjl	%r4,%r0,.Lcheck_notonbb /* Zero index < loaded bytes count ->
					    Reject fits in one vreg;
					    Fill with zeros and proceed
					    with FAST.  */
.Lcheck_onbb2:
	vl	%v17,0(%r3)	/* Load reject, which exceeds loaded bytes.  */
	j	.Lcheck_notonbb /* Check if reject fits in one vreg.  */


	/*
	  Search s for reject in one vreg
	  -------------------------------
	*/
.Lfast:
	/* Complete reject-string in v17 and remaining bytes are zero.  */

	vlbb	%v16,0(%r2),6	/* Load s until next 4k-byte boundary.  */
	lcbb	%r0,0(%r2),6	/* Get bytes to 4k-byte boundary or 16.  */

	vfaezfs	%v18,%v16,%v17,0 /* Find first element in v16
				    unequal to any in v17
				    or first zero element.  */
	vlgvb	%r4,%v18,7	/* Load byte index of found element.  */
	clrjl	%r4,%r0,.Lfast_loop_found2 /* If found index is within loaded
					       bytes, return with found element
					       index (=equal count).  */

	/* Align s to 16 byte.  */
	risbgn	%r4,%r2,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r1,16		/* current_len = 16.  */
	slr	%r1,%r4		/* Compute bytes to 16bytes boundary.  */

	/* Process s in 16byte aligned loop.  */
.Lfast_loop:
	vl	%v16,0(%r1,%r2)	/* Load search-string.  */
	vfaezfs	%v18,%v16,%v17,0 /* Find first element in v16 equal to any
				    in v17 or first zero element.  */
	jno	.Lfast_loop_found

	vl	%v16,16(%r1,%r2)
	vfaezfs	%v18,%v16,%v17,0
	jno	.Lfast_loop_found16

	vl	%v16,32(%r1,%r2)
	vfaezfs	%v18,%v16,%v17,0
	jno	.Lfast_loop_found32

	vl	%v16,48(%r1,%r2)
	vfaezfs	%v18,%v16,%v17,0
	jno	.Lfast_loop_found48

	aghi	%r1,64
	j	.Lfast_loop	/* Loop if no element was unequal to reject
				   and not zero.  */

	/* Found equal or zero element.  */
.Lfast_loop_found48:
	aghi	%r1,16
.Lfast_loop_found32:
	aghi	%r1,16
.Lfast_loop_found16:
	aghi	%r1,16
.Lfast_loop_found:
	vlgvb	%r4,%v18,7	/* Load byte index of found element or zero.  */
.Lfast_loop_found2:
	algrk	%r2,%r1,%r4	/* Add found index to current len.  */
	srlg	%r2,%r2,2	/* Convert byte-count to character-count.  */
	br	%r14



	/*
	  Search s for reject in multiple vregs
	  -------------------------------------
	*/
.Lslow:
	/* Save registers.  */
	vlvgg	%v30,%r6,0
	vlvgp	%v31,%r8,%r9

	/* Reject in v17 without zero.  */
	vlr	%v19,%v17	/* Save first acc-part for a fast reload.  */
	vzero	%v20		/* Zero for preparing acc-vector.  */
	vone	%v24		/* One for checking result of former
				   string-part.  */

	/* Align s to 16 byte.  */
	risbg	%r4,%r2,60,128+63,0 /* Test if s is aligned and
				       %r4 = bits 60-63 'and' 15.   */
	je	.Lslow_loop_str /* If s is aligned, loop aligned.  */
	lghi	%r0,15
	slr	%r0,%r4		/* Compute highest index to load (15-x).  */
	vll	%v16,%r0,0(%r2) /* Load up to 16byte boundary (vll needs
				   highest index, remaining bytes are 0).  */
	ahi	%r0,1		/* Work with loaded byte count.  */
	vzero	%v21		/* Zero out global mask.  */
	lghi	%r5,0		/* Set current len of reject-string to zero.  */
	vfenezf	%v18,%v16,%v16	/* Find zero in current string-part.  */
	lghi	%r8,0		/* There is no zero in first reject-part.  */
	vlgvb	%r6,%v18,7	/* Load byte index of zero or 16 if no zero.  */
	clije	%r6,0,.Lslow_end /* If first element is zero -> return 0.  */
	clr	%r0,%r6		/* cc==1 if loaded byte count < zero-index.  */
	locrl	%r6,%r0		/* Load on cc==1; zero-index = lbc.  */
	j	.Lslow_loop_acc


	/* Process s in 16byte aligned loop.  */
.Lslow_next_str:
	/* Check results of former processed str-part.  */
	vfeef	%v18,%v21,%v24	/* Find first equal match in global mask
				   (ones in element).  */
	vlgvb	%r4,%v18,7	/* Get index of first one (=equal) or 16.  */
	/* Equal-index < min(zero-index, loaded byte count)
	   -> Return pointer to equal element.  */
	clrjl	%r4,%r6,.Lslow_index_found
	/* Zero-index < loaded byte count
	   -> Former str-part was last str-part
	   -> Return null  */
	clrjl	%r6,%r0,.Lslow_end_not_found

	/* All elements are zero (=no match) -> proceed with next str-part.  */
	vlr	%v17,%v19	/* Load first part of reject (no zero).  */
	algfr	%r1,%r0		/* Add loaded byte count to current len.  */

.Lslow_loop_str:
	vl	%v16,0(%r1,%r2)	/* Load search-string.  */
	lghi	%r0,16		/* Loaded byte count is 16.  */
	vzero	%v21		/* Zero out global mask.  */
	lghi	%r5,0		/* Set current len of reject to zero.  */
	vfenezf	%v18,%v16,%v16	/* Find zero in current string-part.  */
	lghi	%r8,0		/* There is no zero in first reject-part.  */
	vlgvb	%r6,%v18,7	/* Load byte index of zero or 16 if no zero.  */
	clije	%r6,0,.Lslow_end /* If first element is zero (end of string)
				     -> Return current length.  */

.Lslow_loop_acc:
	vfaef	%v22,%v16,%v17,4 /* Create matching-mask (1 in mask ->
				    Character matches any rejected character in
				    this reject-string-part) IN=0, RT=1.  */
	vlgvf	%r4,%v22,0	/* Get result of first element.  */
	/* First element is equal to any rejected characters?
	   (All other parts of reject cannot lead to a match before this one)
	   -> Return current len, which is pointing to this element.  */
	clijh	%r4,0,.Lslow_end
	vo	%v21,%v21,%v22	/* Global-mask = global-|matching-mask.  */
	/* Proceed with next acc until end of acc is reached.  */


.Lslow_next_acc:
	clijh	%r8,0,.Lslow_next_str /* There was a zero in last reject-part
					  -> Add found index to current len
					     and end.  */
	vlbb	%v17,16(%r5,%r3),6 /* Load next reject part.  */
	aghi	%r5,16		/* Increment current len of reject-string.  */
	lcbb	%r9,0(%r5,%r3),6 /* Get loaded byte count of reject-string.  */
	jo	.Lslow_next_acc_onbb /* Jump away if reject-string is
					 on block-boundary.  */
.Lslow_next_acc_notonbb:
	vistrfs	%v17,%v17	/* Fill with zeros after first zero.  */
	jo	.Lslow_loop_acc /* No zero found -> no preparation needed.  */

.Lslow_next_acc_prepare_zero:
	/* Zero in reject-part: fill zeros with first-reject-character.  */
	vlgvf	%r8,%v17,0	/* Load first element of reject-part.  */
	clije	%r8,0,.Lslow_next_str /* Process next str-part if first
					  character in this part of reject
					  is a zero.  */
	/* r8>0 -> zero found in this acc-part.  */
	vrepf	%v18,%v17,0	/* Replicate first char accross all chars.  */
	vceqf	%v22,%v20,%v17	/* Create a mask (v22) of null chars
				   by comparing with 0 (v20).  */
	vsel	%v17,%v18,%v17,%v22 /* Replace null chars with first char.  */
	j	.Lslow_loop_acc /* Reject-string part is prepared.  */

.Lslow_next_acc_onbb:
	nill	%r9,65532	/* Recognize only fully loaded characters.  */
	je	.Lslow_next_acc_onbb2 /* Reload vr, if no full wchar_t
					  loaded.  */
	vfenezf	%v18,%v17,%v17	/* Find zero in loaded bytes of reject part.  */
	vlgvb	%r8,%v18,7	/* Load byte index of zero.  */
	clrjl	%r8,%r9,.Lslow_next_acc_notonbb /* Found a zero in loaded bytes
						    -> Prepare vreg.  */
.Lslow_next_acc_onbb2:
	vl	%v17,0(%r5,%r3)	/* Load over boundary ...  */
	lghi	%r8,0		/* r8=0 -> no zero in this part of acc,
				   check for zero is in jump-target.  */
	j	.Lslow_next_acc_notonbb /* ... and search for zero in
					    fully loaded vreg again.  */

.Lslow_end_not_found:
	algfr	%r1,%r6		/* Add zero-index to current len.  */
	j	.Lslow_end
.Lslow_index_found:
	algfr	%r1,%r4		/* Add found index of char to current len.  */
.Lslow_end:
	srlg	%r2,%r1,2	/* Convert byte-count to character-count.  */
	/* Restore registers.  */
	vlgvg	%r6,%v30,0
	vlgvg	%r8,%v31,0
	vlgvg	%r9,%v31,1
	br	%r14
.Lfallback:
	jg	WCSCSPN_C
END(WCSCSPN_Z13)

# if ! HAVE_WCSCSPN_IFUNC
strong_alias (WCSCSPN_Z13, wcscspn)
# endif
#endif
