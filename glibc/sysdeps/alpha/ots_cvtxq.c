/* Software floating-point emulation: float to integer conversion.
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

long
_OtsCvtXQ (long al, long ah, long _round)
{
  FP_DECL_EX;
  FP_DECL_Q(A);
  unsigned long r;
  long s;

  /* If bit 3 is set, then integer overflow detection is requested.  */
  s = _round & 8 ? 1 : -1;
  _round = _round & 3;

  FP_INIT_ROUNDMODE;
  AXP_UNPACK_RAW_Q(A, a);
  FP_TO_INT_Q(r, A, 64, s);
  FP_HANDLE_EXCEPTIONS;

  return r;
}
