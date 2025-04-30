/* Test whether X == Y.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#include <errno.h>
#include <fenv.h>
#include <math.h>
#include <stdbool.h>
#include <fix-fp-int-compare-invalid.h>

int
M_DECL_FUNC (__iseqsig) (FLOAT x, FLOAT y)
{
  /* Comparing <= and >= is sufficient to determine both whether X and
     Y are equal, and whether they are unordered, while raising the
     "invalid" exception if they are unordered.  */
  bool cmp1 = x <= y;
  bool cmp2 = x >= y;
  if (cmp1 && cmp2)
    return 1;
  else if (!cmp1 && !cmp2)
    {
      if (FIX_COMPARE_INVALID)
	__feraiseexcept (FE_INVALID);
      __set_errno (EDOM);
    }
  return 0;
}
