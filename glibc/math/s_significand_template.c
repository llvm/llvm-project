/* Return the mantissa of a floating-point number.
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

/*
 * significand(x) computes just
 * 	scalb(x, (FLOAT) - ilogb(x)),
 * for exercising the fraction-part(F) IEEE 754-1985 test vector.
 */

#include <math.h>
#include <math_private.h>

FLOAT
M_DECL_FUNC (__significand) (FLOAT x)
{
  return M_SUF (__ieee754_scalb) (x,(FLOAT) - M_SUF (__ilogb) (x));
}
declare_mgen_alias (__significand, significand)
