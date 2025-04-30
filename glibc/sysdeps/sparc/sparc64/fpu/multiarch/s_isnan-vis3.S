/* isnan().  sparc64 vis3 version.
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

ENTRY (__isnan_vis3)
	movdtox	%f0, %o0
	sethi	%hi(0x7ff00000), %g1
	sllx	%g1, 32, %g1
	sllx	%o0, 1, %o0
	srlx	%o0, 1, %o0
	sub	%g1, %o0, %o0
	retl
	 srlx	%o0, 63, %o0
END (__isnan_vis3)
