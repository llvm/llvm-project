/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   Contributed by David Mosberger (davidm@cs.arizona.edu).
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* Finds the first bit set in an integer.  Optimized for the Alpha
   architecture.  */

#include <sysdep.h>

	.set noreorder
	.set noat


ENTRY(__ffs)
#ifdef PROF
	ldgp	gp, 0(pv)
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.prologue 1
	zap	$16, 0xF0, $16
	br	$ffsl..ng
#else
	.prologue 0
	zap	$16, 0xF0, $16
	# FALLTHRU
#endif
END(__ffs)

	.align 4
ENTRY(ffsl)
#ifdef PROF
	ldgp	gp, 0(pv)
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.prologue 1
$ffsl..ng:
#else
	.prologue 0
#endif
	not	$16, $1		# e0    :
	ldi	$2, -1		# .. e1 :
	cmpbge	$1, $2, $3	# e0    : bit N == 1 for byte N == 0
	clr	$0		# .. e1 :
	addq	$3, 1, $4	# e0    :
	bic	$4, $3, $3	# e1    : bit N == 1 for first byte N != 0
	and	$3, 0xF0, $4	# e0    :
	and	$3, 0xCC, $5	# .. e1 :
	and	$3, 0xAA, $6	# e0    :
	cmovne	$4, 4, $0	# .. e1 :
	cmovne	$5, 2, $5	# e0    :
	cmovne  $6, 1, $6	# .. e1 :
	addl	$0, $5, $0	# e0    :
	addl	$0, $6, $0	# e1    : $0 == N
	extbl	$16, $0, $1	# e0    : $1 == byte N
	ldi	$2, 1		# .. e1 :
	negq	$1, $3		# e0    :
	and	$3, $1, $3	# e1    : bit N == least bit set of byte N
	and	$3, 0xF0, $4	# e0    :
	and	$3, 0xCC, $5	# .. e1 :
	and	$3, 0xAA, $6	# e0    :
	cmovne	$4, 5, $2	# .. e1 :
	cmovne	$5, 2, $5	# e0    :
	cmovne	$6, 1, $6	# .. e1 :
	s8addl	$0, $2, $0	# e0    : mult byte ofs by 8 and sum
	addl	$5, $6, $5	# .. e1 :
	addl	$0, $5, $0	# e0    :
	nop			# .. e1 :
	cmoveq	$16, 0, $0	# e0    : trap input == 0 case.
	ret			# .. e1 : 18

END(ffsl)

weak_alias (__ffs, ffs)
libc_hidden_def (__ffs)
libc_hidden_builtin_def (ffs)
weak_extern (ffsl)
weak_alias (ffsl, ffsll)
