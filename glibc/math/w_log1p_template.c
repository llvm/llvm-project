/* Wrapper for __log1p that handles setting errno.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
#include <math.h>

FLOAT
M_DECL_FUNC (__w_log1p) (FLOAT x)
{
  if (__glibc_unlikely (islessequal (x, M_LIT (-1.0))))
    {
      if (x == -1)
	__set_errno (ERANGE);
      else
	__set_errno (EDOM);
    }

  return M_SUF (__log1p) (x);
}
declare_mgen_alias (__w_log1p, log1p)
