/* Software floating-point emulation: division.
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

void
_OtsDivX(long al, long ah, long bl, long bh, long _round)
{
  FP_DECL_EX;
  FP_DECL_Q(A); FP_DECL_Q(B); FP_DECL_Q(C);
  AXP_DECL_RETURN_Q(c);

  FP_INIT_ROUNDMODE;
  AXP_UNPACK_Q(A, a);
  AXP_UNPACK_Q(B, b);
  FP_DIV_Q(C, A, B);
  AXP_PACK_Q(c, C);
  FP_HANDLE_EXCEPTIONS;

  AXP_RETURN_Q(c);
}
