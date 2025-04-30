/* isinf().  sparc64 vis3 version.
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

ENTRY (__isinf_vis3)
	movdtox	%f0, %g1
	sethi	%hi(0x7ff00000), %o2
	sllx	%o2, 32, %o2
	sllx	%g1, 1, %o4
	srlx	%o4, 1, %o5
	srax	%g1, 62, %o0
	xor	%o5, %o2, %o3
	retl
	 movrne	%o3, %g0, %o0
END (__isinf_vis3)
