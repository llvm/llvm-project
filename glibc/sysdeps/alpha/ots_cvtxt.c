/* Software floating-point emulation: floating point truncation.
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
#include "double.h"

double
_OtsConvertFloatXT (long al, long ah, long _round)
{
  FP_DECL_EX;
  FP_DECL_Q(A);
  FP_DECL_D(R);
  double r;

  FP_INIT_ROUNDMODE;
  AXP_UNPACK_SEMIRAW_Q(A, a);
#if _FP_W_TYPE_SIZE < 64
  FP_TRUNC(D,Q,2,4,R,A);
#else
  FP_TRUNC(D,Q,1,2,R,A);
#endif
  FP_PACK_SEMIRAW_D(r, R);
  FP_HANDLE_EXCEPTIONS;

  return r;
}
