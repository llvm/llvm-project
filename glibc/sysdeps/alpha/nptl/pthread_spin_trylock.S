/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson  <rth@twiddle.net>, 2003.

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

#include <shlib-compat.h>

#define _ERRNO_H 1
#include <bits/errno.h>

	.text
	.align	4

	.globl	__pthread_spin_trylock
	.ent	__pthread_spin_trylock
__pthread_spin_trylock:
	.frame	$sp, 0, $26, 0
	.prologue 0

0:	ldl_l	$1, 0($16)
	lda	$2, 1
	lda	$0, EBUSY
	bne	$1, 1f

	stl_c	$2, 0($16)
	beq	$2, 2f
	mb
	lda	$0, 0

1:	ret
2:	br	0b

	.end	__pthread_spin_trylock
versioned_symbol (libc, __pthread_spin_trylock, pthread_spin_trylock,
		  GLIBC_2_34)

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, __pthread_spin_trylock, pthread_spin_trylock,
	       GLIBC_2_2)
#endif
