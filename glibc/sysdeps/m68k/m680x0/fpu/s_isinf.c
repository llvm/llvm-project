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
#include "mathimpl.h"

#ifndef FUNC
#define FUNC isinf
#endif
#ifndef float_type
#define float_type double
#endif

#define __CONCATX(a,b) __CONCAT(a,b)

int
__CONCATX(__,FUNC) (float_type x)
{
  return __m81_u(__CONCATX(__,FUNC))(x);
}

#define hidden_defx(a) hidden_def(a)
hidden_defx(__CONCATX(__,FUNC))
weak_alias (__CONCATX(__,FUNC), FUNC)
