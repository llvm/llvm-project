/* Software floating-point emulation.
   Return (long double)(a)
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
#include "double.h"
#include "quad.h"

long double _Q_dtoq(const double a)
{
  FP_DECL_EX;
  FP_DECL_D(A);
  FP_DECL_Q(C);
  long double c;

  FP_INIT_ROUNDMODE;
  FP_UNPACK_RAW_D(A, a);
#if _FP_W_TYPE_SIZE < 64
  FP_EXTEND(Q,D,4,2,C,A);
#else
  FP_EXTEND(Q,D,2,1,C,A);
#endif
  FP_PACK_RAW_Q(c, C);
  FP_HANDLE_EXCEPTIONS;
  return c;
}
