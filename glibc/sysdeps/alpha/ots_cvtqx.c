/* Software floating-point emulation: signed integer to float conversion.
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

/* Should never actually be used, since we've more bits of precision
   than the incomming long, but needed for linkage.  */
#undef FP_ROUNDMODE
#define FP_ROUNDMODE  FP_RND_ZERO

void
_OtsCvtQX (long a)
{
  FP_DECL_EX;
  FP_DECL_Q(C);
  AXP_DECL_RETURN_Q(c);

  FP_FROM_INT_Q(C, a, 64, unsigned long);
  AXP_PACK_RAW_Q(c, C);
  AXP_RETURN_Q(c);
}
