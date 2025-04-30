/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
#undef ret

ENTRY (__copysign)
{
	fmerge.s fret0 = farg1, farg0
	br.ret.sptk.many rp
}
END (__copysign)

strong_alias (__copysign, __copysignf)
strong_alias (__copysign, __copysignl)

strong_alias (__copysign, __libm_copysign)
strong_alias (__copysign, __libm_copysignf)
strong_alias (__copysign, __libm_copysignl)

libm_alias_double (__copysign, copysign)
libm_alias_float (__copysign, copysign)
libm_alias_ldouble (__copysign, copysign)
