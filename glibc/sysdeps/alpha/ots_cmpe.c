/* Software floating-point emulation: comparison.
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

#include "local-soft-fp.h"

static long
internal_compare (long al, long ah, long bl, long bh)
{
  FP_DECL_EX;
  FP_DECL_Q(A); FP_DECL_Q(B);
  long r;

  AXP_UNPACK_RAW_Q(A, a);
  AXP_UNPACK_RAW_Q(B, b);
  FP_CMP_Q (r, A, B, 2, 2);

  FP_HANDLE_EXCEPTIONS;

  return r;
}

long
_OtsLssX (long al, long ah, long bl, long bh)
{
  long r = internal_compare (al, ah, bl, bh);
  if (r == 2)
    return -1;
  else
    return r < 0;
}

long
_OtsLeqX (long al, long ah, long bl, long bh)
{
  long r = internal_compare (al, ah, bl, bh);
  if (r == 2)
    return -1;
  else
    return r <= 0;
}

long
_OtsGtrX (long al, long ah, long bl, long bh)
{
  long r = internal_compare (al, ah, bl, bh);
  if (r == 2)
    return -1;
  else
    return r > 0;
}

long
_OtsGeqX (long al, long ah, long bl, long bh)
{
  long r = internal_compare (al, ah, bl, bh);
  if (r == 2)
    return -1;
  else
    return r >= 0;
}
