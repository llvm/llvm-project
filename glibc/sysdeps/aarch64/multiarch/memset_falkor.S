/* Memset for falkor.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.

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

#include <sysdep.h>
#include <memset-reg.h>

/* Reading dczid_el0 is expensive on falkor so move it into the ifunc
   resolver and assume ZVA size of 64 bytes.  The IFUNC resolver takes care to
   use this function only when ZVA is enabled.  */

#if IS_IN (libc)
.macro zva_macro
	.p2align 4
	/* Write the first and last 64 byte aligned block using stp rather
	   than using DC ZVA.  This is faster on some cores.  */
	str	q0, [dst, 16]
	stp	q0, q0, [dst, 32]
	bic	dst, dst, 63
	stp	q0, q0, [dst, 64]
	stp	q0, q0, [dst, 96]
	sub	count, dstend, dst	/* Count is now 128 too large.	*/
	sub	count, count, 128+64+64	/* Adjust count and bias for loop.  */
	add	dst, dst, 128
1:	dc	zva, dst
	add	dst, dst, 64
	subs	count, count, 64
	b.hi	1b
	stp	q0, q0, [dst, 0]
	stp	q0, q0, [dst, 32]
	stp	q0, q0, [dstend, -64]
	stp	q0, q0, [dstend, -32]
	ret
.endm

# define ZVA_MACRO zva_macro
# define MEMSET __memset_falkor
# include <sysdeps/aarch64/memset.S>
#endif
