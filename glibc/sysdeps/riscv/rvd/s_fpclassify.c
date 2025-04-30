/* fpclassify().  RISC-V version.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <math.h>
#include <fenv_private.h>

int
__fpclassify (double x)
{
  int cls = _FCLASS (x);
  if (__builtin_expect (cls & _FCLASS_NORM, _FCLASS_NORM))
    return FP_NORMAL;
  if (__builtin_expect (cls & _FCLASS_ZERO, _FCLASS_ZERO))
    return FP_ZERO;
  if (__builtin_expect (cls & _FCLASS_SUBNORM, _FCLASS_SUBNORM))
    return FP_SUBNORMAL;
  if (__builtin_expect (cls & _FCLASS_INF, _FCLASS_INF))
    return FP_INFINITE;
  return FP_NAN;
}
libm_hidden_def (__fpclassify)
