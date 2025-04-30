/* Round argument to nearest integral value according to current rounding
   direction.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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
#include <libm-alias-double.h>

	.text
ENTRY(__llrint)
	fldl	4(%esp)
	subl	$8, %esp
	cfi_adjust_cfa_offset (8)
	fistpll	(%esp)
	fwait
	popl	%eax
	cfi_adjust_cfa_offset (-4)
	popl	%edx
	cfi_adjust_cfa_offset (-4)
	ret
END(__llrint)
libm_alias_double (__llrint, llrint)
