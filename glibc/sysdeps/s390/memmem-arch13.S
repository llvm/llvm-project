/* Vector optimized 32/64 bit S/390 version of memmem.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <ifunc-memmem.h>
#if HAVE_MEMMEM_ARCH13
# include "sysdep.h"
# include "asm-syntax.h"
	.text

/* void *memmem(const void *haystack=r2, size_t haystacklen=r3,
		const void *needle=r4, size_t needlelen=r5);
   Locate a substring.  */
ENTRY(MEMMEM_ARCH13)
	.machine "arch13"
	.machinemode "zarch_nohighgprs"
# if ! defined __s390x__
	llgfr	%r3,%r3
	llgfr	%r5,%r5
	llgfr	%r4,%r4
	llgfr	%r2,%r2
# endif /* ! defined __s390x__ */
	clgrjl	%r3,%r5,.Lend_no_match	/* Haystack < needle?  */

	/* Jump to fallback if needle > 9.  See also strstr-arch13.S.  */
# if ! HAVE_MEMMEM_Z13
#  error The arch13 variant of memmem needs the z13 variant of memmem!
# endif
	clgfi	%r5,9
	jh	MEMMEM_Z13

	aghik	%r0,%r5,-1		/* vll needs highest index.  */
	bc	4,0(%r14)		/* cc==1: return if needle-len == 0.  */
	vll	%v18,%r0,0(%r4)		/* Load needle.  */
	vlvgb	%v19,%r5,7		/* v19[7] contains length of needle.  */

	clgijh	%r3,16,.Lhaystack_larger_16
.Lhaystack_smaller_16_on_bb:
	aghik	%r0,%r3,-1		/* vll needs highest index.  */
	vll	%v16,%r0,0(%r2)		/* Load haystack.  */
.Lhaystack_smaller_16:
	sgr	%r3,%r5			/* r3 = largest valid match-index.  */
	jl	.Lend_no_match		/* Haystack-len < needle-len?  */
	vstrs	%v20,%v16,%v18,%v19,0,0
	/* Vector string search without zero search where v20 will contain
	   the index of a partial/full match or 16 (index is named k).
	   cc=0 (no match; k=16): .Lend_no_match
	   cc=1 (only available with zero-search): Ignore
	   cc=2 (full match; k<16): Needle found, but could be beyond haystack!
	   cc=3 (partial match; k<16): Always at end of v16 and thus beyond!  */
	brc	9,.Lend_no_match	/* Jump away if cc == 0 || cc == 3.  */
	vlgvb	%r1,%v20,7
	/* Verify that the full-match (cc=2) is valid!  */
	clgrjh	%r1,%r3,.Lend_no_match	/* Jump away if match is beyond.  */
	la	%r2,0(%r1,%r2)
	br	%r14
.Lend_no_match:
	lghi	%r2,0
	br	%r14

.Lhaystack_larger_16:
	vl	%v16,0(%r2)
	lghi	%r1,17
	lay	%r4,-16(%r3,%r2)	/* Boundary for loading with vl.  */
	lay	%r0,-64(%r3,%r2)	/* Boundary for loading with 4xvl.  */
	/* See also strstr-arch13.S:
	   min-skip-partial-match-index = (16 - n_len) + 1  */
	sgr	%r1,%r5
	clgfi	%r3,64			/* Set Boundary to zero ...  */
	la	%r3,0(%r3,%r2)
	locghil	%r0,0			/* ... if haystack < 64bytes.  */
	jh	.Lloop64
.Lloop:
	la	%r2,16(%r2)
	/* Vector string search with zero search.  cc=0 => no match.  */
	vstrs	%v20,%v16,%v18,%v19,0,0
	jne	.Lloop_vstrs_nonzero_cc
	clgrjh	%r2,%r4,.Lhaystack_too_small
.Lloop16:
	vl	%v16,0(%r2)
	la	%r2,16(%r2)
	vstrs	%v20,%v16,%v18,%v19,0,0
	jne	.Lloop_vstrs_nonzero_cc
	clgrjle	%r2,%r4,.Lloop16
.Lhaystack_too_small:
	sgr	%r3,%r2			/* r3 = (haystack + len) - curr_pos  */
	je	.Lend_no_match		/* Remaining haystack is empty.  */
	lcbb	%r0,0(%r2),6
	jo	.Lhaystack_smaller_16_on_bb
	vl	%v16,0(%r2)		/* Load haystack.  */
	j	.Lhaystack_smaller_16

.Lend_match_found:
	vlgvb	%r4,%v20,7
	sgr	%r2,%r1
	la	%r2,0(%r4,%r2)
	br	%r14

.Lloop_vstrs_nonzero_cc32:
	la	%r2,16(%r2)
.Lloop_vstrs_nonzero_cc16:
	la	%r2,16(%r2)
.Lloop_vstrs_nonzero_cc0:
	la	%r2,16(%r2)
.Lloop_vstrs_nonzero_cc:
	lay	%r2,-16(%r1,%r2)	/* Compute next load address.  */
	jh	.Lend_match_found	/* cc == 2 (full match)  */
	clgrjh	%r2,%r4,.Lhaystack_too_small
	vl	%v16,0(%r2)
.Lloop_vstrs_nonzero_cc_loop:
	la	%r2,0(%r1,%r2)
	vstrs	%v20,%v16,%v18,%v19,0,0
	jh	.Lend_match_found
	clgrjh	%r2,%r4,.Lhaystack_too_small
	vl	%v16,0(%r2)		/* Next part of haystack.  */
	jo	.Lloop_vstrs_nonzero_cc_loop
	/* Case: no-match.  */
	clgrjh	%r2,%r0,.Lloop	/* Jump away if haystack has less than 64b.  */
.Lloop64:
	vstrs	%v20,%v16,%v18,%v19,0,0
	jne	.Lloop_vstrs_nonzero_cc0
	vl	%v16,16(%r2)		/* Next part of haystack.  */
	vstrs	%v20,%v16,%v18,%v19,0,0
	jne	.Lloop_vstrs_nonzero_cc16
	vl	%v16,32(%r2)		/* Next part of haystack.  */
	vstrs	%v20,%v16,%v18,%v19,0,0
	jne	.Lloop_vstrs_nonzero_cc32
	vl	%v16,48(%r2)		/* Next part of haystack.  */
	la	%r2,64(%r2)
	vstrs	%v20,%v16,%v18,%v19,0,0
	jne	.Lloop_vstrs_nonzero_cc
	clgrjh	%r2,%r4,.Lhaystack_too_small
	vl	%v16,0(%r2)		/* Next part of haystack.  */
	clgrjle	%r2,%r0,.Lloop64
	j	.Lloop
END(MEMMEM_ARCH13)

# if ! HAVE_MEMMEM_IFUNC
strong_alias (MEMMEM_ARCH13, __memmem)
weak_alias (__memmem, memmem)
# endif

# if MEMMEM_Z13_ONLY_USED_AS_FALLBACK && defined SHARED && IS_IN (libc)
weak_alias (MEMMEM_ARCH13, __GI_memmem)
strong_alias (MEMMEM_ARCH13, __GI___memmem)
# endif
#endif
