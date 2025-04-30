/* signbit().  sparc64 version.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>

ENTRY (__signbit)
	st	%f0, [%sp + STACK_BIAS + 128]
	ld	[%sp + STACK_BIAS + 128], %o0
	retl
	 srl	%o0, 31, %o0
END (__signbit)

/* On 64-bit the double version will also always work for
   long-double-precision since in both cases the word with the
   sign bit in it is passed always in register %f0.  */
strong_alias (__signbit, __signbitl)
hidden_def (__signbitl)
