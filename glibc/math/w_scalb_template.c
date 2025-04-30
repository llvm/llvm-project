/* Wrapper to set errno for scalb.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
   <http://www.gnu.org/licenses/>.  */

/* Only build wrappers from the templates for the types that define the macro
   below.  This macro is set in math-type-macros-<type>.h in sysdeps/generic
   for each floating-point type.  */
#if __USE_WRAPPER_TEMPLATE

#include <errno.h>
#include <math.h>
#include <math_private.h>

/* Wrapper scalb */
FLOAT M_DECL_FUNC (__scalb) (FLOAT x, FLOAT fn)
{
  FLOAT z = M_SUF (__ieee754_scalb) (x, fn);

  if (__glibc_unlikely (!isfinite (z) || z == M_LIT (0.0)))
    {
      if (isnan (z))
	{
	  if (!isnan (x) && !isnan (fn))
	    __set_errno (EDOM);
	}
      else if (isinf (z))
	{
	  if (!isinf (x) && !isinf (fn))
	    __set_errno (ERANGE);
	}
      else
	{
	  /* z == 0.  */
	  if (x != M_LIT (0.0) && !isinf (fn))
	    __set_errno (ERANGE);
	}
    }
  return z;
}

declare_mgen_alias (__scalb, scalb);

#endif /* __USE_WRAPPER_TEMPLATE.  */
