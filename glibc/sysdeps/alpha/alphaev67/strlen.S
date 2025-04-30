/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   Contributed by David Mosberger (davidm@cs.arizona.edu).
   EV67 optimized by Rick Gorton <rick.gorton@alpha-processor.com>.
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

/* Finds length of a 0-terminated string.  */

#include <sysdep.h>

	.arch ev6
	.set noreorder
	.set noat

ENTRY(strlen)
#ifdef PROF
	ldgp	gp, 0(pv)
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.prologue 1
#else
	.prologue 0
#endif

	ldq_u	$1, 0($16)	# L : load first quadword ($16 may be misaligned)
	lda	$2, -1($31)	# E :
	insqh	$2, $16, $2	# U :
	andnot	$16, 7, $0	# E :

	or	$2, $1, $1	# E :
	cmpbge	$31, $1, $2	# E : $2  <- bitmask: bit i == 1 <==> i-th byte == 0
	nop			# E :
	bne	$2, $found	# U :

$loop:	ldq	$1, 8($0)	# L :
	addq	$0, 8, $0	# E : addr += 8
	cmpbge	$31, $1, $2	# E :
	beq	$2, $loop	# U :

$found:
	cttz	$2, $3		# U0 :
	addq	$0, $3, $0	# E :
	subq	$0, $16, $0	# E :
	ret	$31, ($26)	# L0 :

	END(strlen)
libc_hidden_builtin_def (strlen)
