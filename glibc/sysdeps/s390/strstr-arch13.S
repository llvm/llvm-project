/* Vector optimized 32/64 bit S/390 version of strstr.
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

#include <ifunc-strstr.h>
#if HAVE_STRSTR_ARCH13
# include "sysdep.h"
# include "asm-syntax.h"
	.text

/* char *strstr (const char *haystack=r2, const char *needle=r3)
   Locate a substring.  */
ENTRY(STRSTR_ARCH13)
	.machine "arch13"
	.machinemode "zarch_nohighgprs"
	lcbb	%r1,0(%r3),6
	jo	.Lneedle_on_bb	/* Needle on block-boundary?  */
	vl	%v18,0(%r3),6	/* Load needle.  */
	vfenezb %v19,%v18,%v18	/* v19[7] contains the length of needle.  */
.Lneedle_loaded:
	vlgvb	%r4,%v19,7	/* Get index of zero or 16 if not found.  */
	lghi	%r5,17		/* See below: min-skip-partial-match-index.  */
	cgibe	%r4,0,0(%r14)	/* Test if needle is zero and return.  */

	/* The vstrs instruction is able to handle needles up to a length of 16,
	   but then we may have to load the next part of haystack with a
	   small offset.  This will be slow - see examples:
	   haystack =mmmmmmmmmmmmmmmm mmmmmmmmmmmmmmmmmm...mmmmmmmmmmmmmmmmmmma
	   needle   =  mmmmmmmmmmmmmma0
	   => needle_len=15; vstrs reports a partial match; haystack+=2
	   haystack =mmmmmmmmmmmmmmmm mmmmmmmmmmmmmmmmmm...mmmmmmmmmmmmmmmmmmma
	   needle   =        mmmmmmmma0000000
	   => needle_len=9; vstrs reports a partial match; haystack+=8  */
# if ! HAVE_STRSTR_Z13
#  error The arch13 variant of strstr needs the z13 variant of strstr!
# endif
	clgfi	%r4,9
	jh	STRSTR_Z13

	/* In case of a partial match, the vstrs instruction returns the index
	   of the partial match in a vector-register.  Then we have to
	   reload the string at the "current-position plus this index" and run
	   vstrs again in order to determine if it was a full match or no match.
	   Transferring this index from vr to gr, compute the haystack-address
	   and loading with vl is quite slow as all instructions have data
	   dependencies.  Thus we assume, that a partial match is always at the
	   first possible index and just load the next part of haystack from
	   there instead of waiting until the correct index is computed:
	   min-skip-partial-match-index = (16 - n_len) + 1  */
	sgr	%r5,%r4

.Lloop:
	lcbb	%r1,0(%r2),6
	jo	.Lloop_haystack_on_bb	/* Haystack on block-boundary?  */
	vl	%v16,0(%r2)		/* Load next part of haystack.  */
.Lloop_haystack_loaded:
	/* Vector string search with zero search (cc=0 => no match).  */
	vstrs	%v20,%v16,%v18,%v19,0,2
	jne	.Lloop_vstrs_nonzero_cc
	lcbb	%r1,16(%r2),6		/* Next part of haystack.  */
	jo	.Lloop_haystack_on_bb16
	vl	%v16,16(%r2)
	vstrs	%v20,%v16,%v18,%v19,0,2
	jne	.Lloop_vstrs_nonzero_cc16
	lcbb	%r1,32(%r2),6		/* Next part of haystack.  */
	jo	.Lloop_haystack_on_bb32
	vl	%v16,32(%r2)
	vstrs	%v20,%v16,%v18,%v19,0,2
	jne	.Lloop_vstrs_nonzero_cc32
	lcbb	%r1,48(%r2),6		/* Next part of haystack.  */
	jo	.Lloop_haystack_on_bb48
	vl	%v16,48(%r2)
	vstrs	%v20,%v16,%v18,%v19,0,2
	jne	.Lloop_vstrs_nonzero_cc48
	la	%r2,64(%r2)
	j	.Lloop

.Lloop_vstrs_nonzero_cc48:
	la	%r2,16(%r2)
.Lloop_vstrs_nonzero_cc32:
	la	%r2,16(%r2)
.Lloop_vstrs_nonzero_cc16:
	la	%r2,16(%r2)
.Lloop_vstrs_nonzero_cc:
	jh	.Lend_match_found /* cc == 2 (full match)  */
	jl	.Lend_no_match	/* cc == 1 (no match, end of string)  */
	/* cc == 3 (partial match) See above: min-skip-partial-match-index!  */
	lcbb	%r1,0(%r5,%r2),6
	la	%r2,0(%r5,%r2)
	jo	.Lloop_haystack_on_bb
	vl	%v16,0(%r2)
	vstrs	%v20,%v16,%v18,%v19,0,2
.Lloop_vstrs_nonzero_cc_loop:
	jh	.Lend_match_found
	jl	.Lend_no_match
	la	%r2,0(%r5,%r2)
	je	.Lloop
	lcbb	%r1,0(%r2),6		/* Next part of haystack.  */
	jo	.Lloop_haystack_on_bb
	vl	%v16,0(%r2)
	vstrs	%v20,%v16,%v18,%v19,0,2
	jh	.Lend_match_found
	jl	.Lend_no_match
	la	%r2,0(%r5,%r2)
	je	.Lloop
	lcbb	%r1,0(%r2),6		/* Next part of haystack.  */
	jo	.Lloop_haystack_on_bb
	vl	%v16,0(%r2)
	vstrs	%v20,%v16,%v18,%v19,0,2
	jh	.Lend_match_found
	jl	.Lend_no_match
	la	%r2,0(%r5,%r2)
	je	.Lloop
	lcbb	%r1,0(%r2),6		/* Next part of haystack.  */
	jo	.Lloop_haystack_on_bb
	vl	%v16,0(%r2)
	vstrs	%v20,%v16,%v18,%v19,0,2
	j	.Lloop_vstrs_nonzero_cc_loop

.Lend_no_match:
	lghi	%r2,0
	br	%r14
.Lend_match_found:
	vlgvb	%r4,%v20,7
	la	%r2,0(%r4,%r2)
	br	%r14

.Lloop_haystack_on_bb48:
	la	%r2,16(%r2)
.Lloop_haystack_on_bb32:
	la	%r2,16(%r2)
.Lloop_haystack_on_bb16:
	la	%r2,16(%r2)
.Lloop_haystack_on_bb:
	/* Haystack located on page-boundary.  */
	ahi	%r1,-1		/* vll needs highest index instead of count.  */
	vll	%v16,%r1,0(%r2)
	vlvgb	%v21,%r1,7
	vfenezb	%v17,%v16,%v16	/* Search zero in loaded haystack bytes.  */
	veclb	%v17,%v21		/* Zero index <= loaded byte index?  */
	jle	.Lloop_haystack_loaded	/* -> v16 contains full haystack.  */
	vl	%v16,0(%r2)	/* Load haystack beyond page boundary.  */
	j	.Lloop_haystack_loaded

.Lneedle_on_bb:
	/* Needle located on page-boundary.  */
	ahi	%r1,-1		/* vll needs highest index instead of count.  */
	vll	%v18,%r1,0(%r3)
	vlvgb	%v21,%r1,7
	vfenezb	%v19,%v18,%v18	/* Search zero in loaded needle bytes.  */
	veclb	%v19,%v21	/* Zero index <= max loaded byte index?  */
	jle	.Lneedle_loaded	/* -> v18 contains full needle.  */
	vl	%v18,0(%r3)	/* Load needle beyond page boundary.  */
	vfenezb	%v19,%v18,%v18
	j	.Lneedle_loaded
END(STRSTR_ARCH13)

# if ! HAVE_STRSTR_IFUNC
strong_alias (STRSTR_ARCH13, strstr)
# endif

# if STRSTR_Z13_ONLY_USED_AS_FALLBACK && defined SHARED && IS_IN (libc)
strong_alias (STRSTR_ARCH13, __GI_strstr)
# endif
#endif
