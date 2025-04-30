/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
#include <math_private.h>
#include "mathimpl.h"
#include <libm-alias-finite.h>

#ifndef	FUNC
# define FUNC __ieee754_acos
# define FUNC_FINITE __acos
#endif
#ifndef float_type
# define float_type double
#endif

float_type
FUNC (float_type x)
{
  return __m81_u(FUNC)(x);
}
#ifdef FUNC_FINITE
libm_alias_finite (FUNC, FUNC_FINITE)
#endif
