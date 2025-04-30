/* Wrapper to set errno for pow.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* Only build wrappers from the templates for the types that define the macro
   below.  This macro is set in math-type-macros-<type>.h in sysdeps/generic
   for each floating-point type.  */
#if __USE_WRAPPER_TEMPLATE

# include <errno.h>
# include <fenv.h>
# include <math.h>
# include <math_private.h>

FLOAT
M_DECL_FUNC (__pow) (FLOAT x, FLOAT y)
{
  FLOAT z = M_SUF (__ieee754_pow) (x, y);
  if (__glibc_unlikely (!isfinite (z)))
    {
      if (isfinite (x) && isfinite (y))
	{
	  if (isnan (z))
	    /* Domain error: pow(x<0,y=non-integer).  */
	    __set_errno (EDOM);
	  else
	    /* Pole error: pow(x=0,y<0).  Or overflow.  */
	    __set_errno (ERANGE);
	}
    }
  else if (__glibc_unlikely (z == 0) && isfinite (x) && x != 0 && isfinite (y))
    /* Underflow.  */
    __set_errno (ERANGE);
  return z;
}
declare_mgen_alias (__pow, pow)

#endif /* __USE_WRAPPER_TEMPLATE.  */
