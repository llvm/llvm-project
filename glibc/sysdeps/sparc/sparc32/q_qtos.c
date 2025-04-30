/* Software floating-point emulation.
   Return (float)a
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
#include "single.h"
#include "quad.h"

float _Q_qtos(const long double a)
{
  FP_DECL_EX;
  FP_DECL_Q(A);
  FP_DECL_S(R);
  float r;

  FP_INIT_ROUNDMODE;
  FP_UNPACK_SEMIRAW_Q(A, a);
#if _FP_W_TYPE_SIZE < 64
  FP_TRUNC(S,Q,1,4,R,A);
#else
  FP_TRUNC(S,Q,1,2,R,A);
#endif
  FP_PACK_SEMIRAW_S(r, R);
  FP_HANDLE_EXCEPTIONS;

  return r;
}
