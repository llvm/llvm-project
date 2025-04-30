/* Return classification value corresponding to argument.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997 and
		  Jakub Jelinek <jj@ultra.linux.cz>, 1999.

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

#include <math.h>

#include <math_private.h>


int
__fpclassifyl (_Float128 x)
{
  uint64_t hx, lx;
  int retval = FP_NORMAL;

  GET_LDOUBLE_WORDS64 (hx, lx, x);
  lx |= (hx & 0x0000ffffffffffffLL);
  hx &= 0x7fff000000000000LL;
  if ((hx | lx) == 0)
    retval = FP_ZERO;
  else if (hx == 0)
    retval = FP_SUBNORMAL;
  else if (hx == 0x7fff000000000000LL)
    retval = lx != 0 ? FP_NAN : FP_INFINITE;

  return retval;
}
libm_hidden_def (__fpclassifyl)
