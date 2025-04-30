/* Round argument to nearest integral value according to current rounding
   direction.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.d>, 2002.

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
#include <libm-alias-float.h>

	.text
ENTRY(__llrintf)
	cvtss2si %xmm0,%rax
	ret
END(__llrintf)
libm_alias_float (__llrint, llrint)
#ifndef __ILP32__
strong_alias (__llrintf, __lrintf)
libm_alias_float (__llrint, lrint)
#endif
