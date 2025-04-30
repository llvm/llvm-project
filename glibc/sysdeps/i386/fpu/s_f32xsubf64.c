/* Subtract _Float64 values, converting the result to _Float32x.  i386 version.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <math.h>
#include <fpu_control.h>
#include <math-narrow-eval.h>
#include <math-narrow.h>

_Float32x
__f32xsubf64 (_Float64 x, _Float64 y)
{
  /* To avoid double rounding, set double precision for the subtraction.
     math_narrow_eval is still needed to eliminate excess range in the
     case of overflow.  If the result of the subtraction is in the
     subnormal range for double, it is exact, so no issues of double
     rounding for subnormals arise.  */
  fpu_control_t cw, cw_double;
  _FPU_GETCW (cw);
  cw_double = (cw & ~_FPU_EXTENDED) | _FPU_DOUBLE;
  _FPU_SETCW (cw_double);
  _Float32x ret = math_narrow_eval (x - y);
  _FPU_SETCW (cw);
  CHECK_NARROW_SUB (ret, x, y);
  return ret;
}
libm_alias_float32x_float64 (sub)
