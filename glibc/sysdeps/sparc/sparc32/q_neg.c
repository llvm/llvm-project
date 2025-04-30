/* Software floating-point emulation.
   Return !a
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include "soft-fp.h"
#include "quad.h"

long double _Q_neg(const long double a)
{
  union {
    long double	ldbl;
    UWtype	words[4];
  } c;

  c.ldbl = a;

#if (__BYTE_ORDER == __BIG_ENDIAN)
  c.words[0] ^= (((UWtype)1) << (W_TYPE_SIZE - 1));
#elif (__BYTE_ORDER == __LITTLE_ENDIAN) && (W_TYPE_SIZE == 64)
  c.words[1] ^= (((UWtype)1) << (W_TYPE_SIZE - 1));
#elif (__BYTE_ORDER == __LITTLE_ENDIAN) && (W_TYPE_SIZE == 32)
  c.words[3] ^= (((UWtype)1) << (W_TYPE_SIZE - 1));
#else
  FP_DECL_Q(A); FP_DECL_Q(C);

  FP_UNPACK_RAW_Q(A, a);
  FP_NEG_Q(C, A);
  FP_PACK_RAW_Q(c.ldbl, C);
#endif
  return c.ldbl;
}
