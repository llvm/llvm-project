/* Return positive difference between arguments.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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
#include <math.h>
#include <math-narrow-eval.h>

FLOAT
M_DECL_FUNC (__fdim) (FLOAT x, FLOAT y)
{
  if (islessequal (x, y))
    return 0;

  FLOAT r = math_narrow_eval (x - y);
  if (isinf (r) && !isinf (x) && !isinf (y))
    __set_errno (ERANGE);

  return r;
}
declare_mgen_alias (__fdim, fdim);
