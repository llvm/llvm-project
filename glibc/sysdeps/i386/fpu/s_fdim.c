/* Return positive difference between arguments.  i386 version.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
#include <fpu_control.h>
#include <math.h>
#include <math-narrow-eval.h>
#include <libm-alias-double.h>

double
__fdim (double x, double y)
{
  if (islessequal (x, y))
    return 0.0;

  /* To avoid double rounding, set double precision for the
     subtraction.  math_narrow_eval is still needed to eliminate
     excess range in the case of overflow.  If the result of the
     subtraction is in the subnormal range for double, it is exact, so
     no issues of double rounding for subnormals arise.  */
  fpu_control_t cw, cw_double;
  _FPU_GETCW (cw);
  cw_double = (cw & ~_FPU_EXTENDED) | _FPU_DOUBLE;
  _FPU_SETCW (cw_double);
  double r = math_narrow_eval (x - y);
  _FPU_SETCW (cw);
  if (isinf (r) && !isinf (x) && !isinf (y))
    __set_errno (ERANGE);

  return r;
}
libm_alias_double (__fdim, fdim)
