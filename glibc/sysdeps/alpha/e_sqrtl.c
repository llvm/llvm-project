/* long double square root in software floating-point emulation.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson (rth@cygnus.com) and
		  Jakub Jelinek (jj@ultra.linux.cz).

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

#include <stdlib.h>
#include <soft-fp.h>
#include <quad.h>
#include <shlib-compat.h>

long double
__ieee754_sqrtl (const long double a)
{
  FP_DECL_EX;
  FP_DECL_Q(A); FP_DECL_Q(C);
  long double c;
  long _round = 4;	/* dynamic rounding */

  FP_INIT_ROUNDMODE;
  FP_UNPACK_Q(A, a);
  FP_SQRT_Q(C, A);
  FP_PACK_Q(c, C);
  FP_HANDLE_EXCEPTIONS;
  return c;
}

/* ??? We forgot to add this symbol in 2.15.  Getting this into 2.18 isn't as
   straight-forward as just adding the alias, since a generic Versions file
   includes the 2.15 version and the linker uses the first one it sees.  */
#if SHLIB_COMPAT (libm, GLIBC_2_15, GLIBC_2_18)
compat_symbol (libm, __ieee754_sqrtl, __sqrtl_finite, GLIBC_2_18);
#endif
