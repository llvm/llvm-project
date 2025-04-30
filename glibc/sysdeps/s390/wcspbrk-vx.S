/* Vector optimized 32/64 bit S/390 version of wcspbrk.
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

#include <ifunc-wcspbrk.h>
#if HAVE_WCSPBRK_Z13

# include "sysdep.h"
# include "asm-syntax.h"

	.text

/* wchar_t *wcspbrk (const wchar_t *s, const wchar_t * accept)
   The  wcspbrk()  function locates the first occurrence in the string s
   of any of the characters in the string accept and returns a pointer
   to that character or NULL if not found.

   This method checks the length of accept string. If it fits entirely
   in one vector register, a fast algorithm is used, which does not need
   to check multiple parts of accept-string. Otherwise a slower full
   check of accept-string is used.

   register overview:
   r3:  pointer to start of accept-string
   r2:  pointer to start of search-string
   r0:  loaded byte count of vlbb search-string (32bit unsigned)
   r4:  found byte index (32bit unsigned)
   r1:  current return len (64bit unsigned)
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
   v24: one for result-checking of former string-part
   v30, v31: for re-/storing registers r6, r8, r9
   r5:  current len of accept-string
   r6:  zero-index in search-string or 16 if no zero
	or min(zero-index, loaded byte count)
   r8:  >0, if former accept-string-part contains a zero,
	otherwise =0;
   r9:  loaded byte count of vlbb accept-string
*/
ENTRY(WCSPBRK_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"

	tmll	%r2,3		/* Test if s is 4-byte aligned?  */
	jne	.Lfallback	/* And use common-code variant if not.  */

	/*
	  Check if accept-string fits in one vreg:
	  ----------------------------------------
	*/
	vlbb	%v17,0(%r3),6	/* Load accept.  */
	lcbb	%r0,0(%r3),6
	jo	.Lcheck_onbb	/* Special case if accept lays
				   on block-boundary.  */

.Lcheck_notonbb:
	lghi	%r1,0		/* Zero out current len.  */
	vlgvf	%r0,%v17,0	/* Get first element.  */
	clije	%r0,0,.Lfast_end_null /* Return null if accept is empty.  */

	vistrfs	%v17,%v17	/* Fill with zeros after first zero.  */
	je	.Lfast		/* Zero found -> accept fits in one vreg.  */
	j	.Lslow		/* No zero -> accept exceeds one vreg  */


.Lcheck_onbb:
	/* Accept lays on block-boundary.  */
	nill	%r0,65532	/* Recognize only fully loaded characters.  */
	je	.Lcheck_onbb2	/* Reload vr, if we loaded no full wchar_t.  */
	vfenezf	%v18,%v17,%v17	/* Search zero in loaded accept bytes.  */
	vlgvb	%r4,%v18,7	/* Get index of zero or 16 if not found.  */
	clrjl	%r4,%r0,.Lcheck_notonbb /* Zero index < loaded bytes count ->
					    accept fits in one vreg;
					    Fill with zeros and proceed
					    with FAST.  */
.Lcheck_onbb2:
	vl	%v17,0(%r3)	/* Load accept, which exceeds loaded bytes.  */
	j	.Lcheck_notonbb /* Check if accept fits in one vreg.  */


	/*
	  Search s for accept in one vreg
	  -------------------------------
	*/
.Lfast:
	/* Complete accept-string in v17 and remaining bytes are zero.  */

	vlbb	%v16,0(%r2),6	/* Load s until next 4k-byte boundary.  */
	lcbb	%r0,0(%r2),6	/* Get bytes to 4k-byte boundary or 16.  */

	vfaezfs	%v18,%v16,%v17,0 /* Find first element in v16 unequal to any
				    in v17 or first zero element.  */
	vlgvb	%r4,%v18,7	/* Load byte index of found element.  */
	/* If found index is within loaded bytes, return with found
	   element index (=equal count).  */
	clrjl	%r4,%r0,.Lfast_loop_found2

	/* Align s to 16 byte.  */
	risbgn	%r4,%r2,60,128+63,0 /* %r3 = bits 60-63 of %r2 'and' 15.  */
	lghi	%r1,16		/* current_len = 16.  */
	slr	%r1,%r4		/* Compute bytes to 16bytes boundary.  */

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
	j	.Lfast_loop	/* Loop if no element was unequal to accept
				   and not zero.  */

	/* Found equal or zero element.  */
.Lfast_loop_found48:
	aghi	%r1,16
.Lfast_loop_found32:
	aghi	%r1,16
.Lfast_loop_found16:
	aghi	%r1,16
.Lfast_loop_found:
	vlgvb	%r4,%v18,7	/* Load byte index of found element.  */
.Lfast_loop_found2:
	srlg	%r5,%r4,2	/* Convert byte-index to character-index.  */
	vlgvf	%r0,%v16,0(%r5)	/* Get found element.  */
	clije	%r0,0,.Lfast_end_null /* Return null if no accept-char found */
	algfr	%r1,%r4		/* Add found index of char to current len.  */
	la	%r2,0(%r1,%r2)	/* And return pointer to first equal char.  */
	br	%r14

.Lfast_end_null:
	lghi	%r2,0		/* Return null if no character is equal.  */
	br	%r14




	/*
	  Search s for accept in multiple vregs
	  -------------------------------------
	*/
.Lslow:
	/* Save registers.  */
	vlvgg	%v30,%r6,0
	vlvgp	%v31,%r8,%r9

	/* Accept in v17 without zero */
	vlr	%v19,%v17	/* Save first acc-part for a fast reload.  */
	vzero	%v20		/* Zero for preparing acc-vector.  */
	vone	%v24		/* One for checking result of former string.  */

	/* Align s to 16 byte.  */
	risbg	%r4,%r2,60,128+63,0 /* Test if s is aligned and
				       %r4 = bits 60-63 'and' 15.  */
	je	.Lslow_loop_str /* If s is aligned, loop aligned.  */
	lghi	%r0,15
	slr	%r0,%r4		/* Compute highest index to load (15-x).  */
	vll	%v16,%r0,0(%r2) /* Load up to 16byte boundary;
				   needs highest index, left bytes are 0.  */
	ahi	%r0,1		/* Work with loaded byte count.  */
	vzero	%v21		/* Zero out global mask.  */
	lghi	%r5,0		/* Set current len of accept-string to zero.  */
	vfenezf	%v18,%v16,%v16	/* Find zero in current string-part.  */
	lghi	%r8,0		/* There is no zero in first accept-part.  */
	vlgvb	%r6,%v18,7	/* Load byte index of zero or 16 if no zero.  */
	clije	%r6,0,.Lslow_end_null /* If first element is zero
					  (end of string) -> return null */
	clr	%r0,%r6		/* cc==1 if loaded byte count < zero-index.  */
	locrl	%r6,%r0		/* Load on cc==1; zero-index = lbc.  */
	j	.Lslow_loop_acc


	/* Process s in 16byte aligned loop.  */
.Lslow_next_str:
	/* Check results of former processed str-part.  */
	vfeef	%v18,%v21,%v24	/* Find first equal match in global mask
				   (ones in element).  */
	vlgvb	%r4,%v18,7	/* Get index of first one (=equal)
				   or 16 if no match.  */
	/* Equal-index < min(zero-index, loaded byte count)
	   -> return pointer to equal element.  */
	clrjl	%r4,%r6,.Lslow_index_found
	/* Zero-index < loaded byte count
	   -> former str-part was last str-part
	   -> return null */
	clrjl	%r6,%r0,.Lslow_end_null
	/* All elements are zero (=no match) -> proceed with next str-part.  */

	vlr	%v17,%v19	/* Load first part of accept (no zero).  */
	algfr	%r1,%r0		/* Add loaded byte count to current len.  */

.Lslow_loop_str:
	vl	%v16,0(%r1,%r2)	/* Load search-string */
	lghi	%r0,16		/* Loaded byte count is 16.  */
	vzero	%v21		/* Zero out global mask.  */
	lghi	%r5,0		/* Set current len of accept to zero.  */
	vfenezf	%v18,%v16,%v16	/* Find zero in current string-part.  */
	lghi	%r8,0		/* There is no zero in first accept-part.  */
	vlgvb	%r6,%v18,7	/* Load byte index of zero or 16 if no zero.  */
	clije	%r6,0,.Lslow_end_null /* If first element is zero
					  (end of string) -> return null.  */

.Lslow_loop_acc:
	vfaef	%v22,%v16,%v17,4 /* Create matching-mask (1 in mask ->
				    Character matches any accepted character in
				    this accept-string-part) IN=0, RT=1.  */
	vlgvf	%r4,%v22,0	/* Get result of first element.  */
	/* First element is equal to any accepted characters
	   (all other parts of accept cannot lead to a match before this one)
	   -> current len is pointing to first element
	   -> return found */
	clijh	%r4,0,.Lslow_end_found
	vo	%v21,%v21,%v22	/* Global-mask = global-|matching-mask.  */
	/* Proceed with next acc until end of acc is reached.  */


.Lslow_next_acc:
	clijh	%r8,0,.Lslow_next_str /* There was a zero in the last acc-part
					  -> add index to current len and
					     end.  */
	vlbb	%v17,16(%r5,%r3),6 /* Load next accept part.  */
	aghi	%r5,16		/* Increment current len of accept-string.  */
	lcbb	%r9,0(%r5,%r3),6 /* Get loaded byte count of accept-string.  */
	jo	.Lslow_next_acc_onbb /* Jump away ifaccept-string is
					 on block-boundary.  */
.Lslow_next_acc_notonbb:
	vistrfs	%v17,%v17	/* Fill with zeros after first zero.  */
	jo	.Lslow_loop_acc /* No zero found -> no preparation needed.  */

.Lslow_next_acc_prepare_zero:
	/* Zero in accept-part: fill zeros with first-accept-character.  */
	vlgvf	%r8,%v17,0	/* Load first element of acc-part.  */
	clije	%r8,0,.Lslow_next_str /* Proceed with next string-part,
					  If first char in this part of accept
					  is a zero.  */
	/* r8>0 -> zero found in this acc-part.  */
	vrepf	%v18,%v17,0	/* Replicate first char accross all chars.  */
	vceqf	%v22,%v20,%v17	/* Create a mask (v22) of null chars
				   by comparing with 0 (v20).  */
	vsel	%v17,%v18,%v17,%v22 /* Replace null chars with first char.  */
	j	.Lslow_loop_acc /* Accept part is prepared -> process.  */

.Lslow_next_acc_onbb:
	nill	%r9,65532	/* Recognize only fully loaded characters.  */
	je	.Lslow_next_acc_onbb2 /* Reload vr, if no full wchar_t.  */
	vfenezf	%v18,%v17,%v17	/* Find zero in loaded bytes of accept part.  */
	vlgvb	%r8,%v18,7	/* Load byte index of zero.  */
	clrjl	%r8,%r9,.Lslow_next_acc_notonbb /* Found a zero in loaded bytes
						    -> Prepare vreg.  */
.Lslow_next_acc_onbb2:
	vl	%v17,0(%r5,%r3)	/* Load over boundary ...  */
	lghi	%r8,0		/* r8=0 -> no zero in this part of acc,
				   check for zero is in jump-target.  */
	j	.Lslow_next_acc_notonbb /* ... and search for zero in
					    fully loaded vreg again.  */

.Lslow_end_null:
	lghi	%r1,0		/* Return null if no character is equal.  */
	j	.Lslow_end

.Lslow_loop_found:
	vlgvb	%r4,%v18,7	/* Load byte index of found element.  */
	srlg	%r5,%r4,2	/* Convert byte-index to character-index.  */
	vlgvf	%r0,%v16,0(%r5)	/* Get found element.  */
	clije	%r0,0,.Lslow_end_null /* Return null if no acc-char found.  */

.Lslow_index_found:
	algfr	%r1,%r4		/* Add found index of char to current len.  */
.Lslow_end_found:
	la	%r1,0(%r1,%r2)	/* And return pointer to first equal char.  */

.Lslow_end:
	/* Restore registers.  */
	vlgvg	%r6,%v30,0
	vlgvg	%r8,%v31,0
	vlgvg	%r9,%v31,1
	lgr	%r2,%r1
	br	%r14
.Lfallback:
	jg	WCSPBRK_C
END(WCSPBRK_Z13)

# if ! HAVE_WCSPBRK_IFUNC
strong_alias (WCSPBRK_Z13, wcspbrk)
# endif

# if defined HAVE_S390_MIN_Z13_ZARCH_ASM_SUPPORT \
	&& defined SHARED && IS_IN (libc)
strong_alias (WCSPBRK_Z13, __GI_wcspbrk)
# endif
#endif
