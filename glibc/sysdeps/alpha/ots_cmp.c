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
internal_equality (long al, long ah, long bl, long bh, long neq)
{
  FP_DECL_EX;
  FP_DECL_Q(A); FP_DECL_Q(B);
  long r;

  AXP_UNPACK_RAW_Q(A, a);
  AXP_UNPACK_RAW_Q(B, b);

  if ((A_e == _FP_EXPMAX_Q && !_FP_FRAC_ZEROP_2(A))
       || (B_e == _FP_EXPMAX_Q && !_FP_FRAC_ZEROP_2(B)))
    {
      /* EQ and NE signal invalid operation only if either operand is SNaN.  */
      if (FP_ISSIGNAN_Q(A) || FP_ISSIGNAN_Q(B))
	{
	  FP_SET_EXCEPTION(FP_EX_INVALID);
	  FP_HANDLE_EXCEPTIONS;
	}
      return -1;
    }

  r = (A_e == B_e
       && _FP_FRAC_EQ_2 (A, B)
       && (A_s == B_s || (!A_e && _FP_FRAC_ZEROP_2(A))));
  r ^= neq;

  return r;
}

long
_OtsEqlX (long al, long ah, long bl, long bh)
{
  return internal_equality (al, ah, bl, bh, 0);
}

long
_OtsNeqX (long al, long ah, long bl, long bh)
{
  return internal_equality (al, ah, bl, bh, 1);
}
