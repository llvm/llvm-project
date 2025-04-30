/* Implement tan for m68k.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
#include <errno.h>
#include "mathimpl.h"

FLOAT
M_DECL_FUNC (__tan) (FLOAT x)
{
  if (__m81_test (x) & __M81_COND_INF)
    __set_errno (EDOM);
  return __m81_u(M_SUF (__tan)) (x);
}
declare_mgen_alias (__tan, tan)
