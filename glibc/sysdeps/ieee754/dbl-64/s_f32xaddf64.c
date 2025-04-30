/* Add _Float64 values, converting the result to _Float32x.
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

#define daddl __hide_daddl
#include <math.h>
#undef daddl

#include <math-narrow.h>

_Float32x
__f32xaddf64 (_Float64 x, _Float64 y)
{
  NARROW_ADD_TRIVIAL (x, y, _Float32x);
}
libm_alias_float32x_float64 (add)
