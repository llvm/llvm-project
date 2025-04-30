/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   Contributed by Richard Henderson <rth@tamu.edu>, 1996.
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

/* Append a null-terminated string from SRC to DST.  */

#include <sysdep.h>

	.arch ev6
	.set noreorder
	.text

ENTRY(strcat)
	ldgp	gp, 0(pv)
#ifdef PROF
	.set noat
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.set at
#endif
	.prologue 1

	mov	$16, $0		# E : set up return value
	/* Find the end of the string.  */
	ldq_u   $1, 0($16)	# L : load first quadword (a0 may be misaligned)
	lda     $2, -1		# E :
	insqh   $2, $16, $2	# U :

	andnot  $16, 7, $16	# E :
	or      $2, $1, $1	# E :
	cmpbge  $31, $1, $2	# E : bits set iff byte == 0
	bne     $2, $found	# U :

$loop:	ldq     $1, 8($16)	# L :
	addq    $16, 8, $16	# E :
	cmpbge  $31, $1, $2	# E :
	beq     $2, $loop	# U :

$found:	cttz	$2, $3		# U0 :
	addq	$16, $3, $16	# E :
	/* Now do the append.  */
	mov	$26, $23	# E :
	jmp	$31, __stxcpy	# L0 :

	END(strcat)
libc_hidden_builtin_def (strcat)
