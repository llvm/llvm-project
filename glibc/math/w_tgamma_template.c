/* Wrapper to set errno for tgamma.
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

#define M_CALL_FUNC_X(x) x ## _r
#define M_CALL_FUNC(x) M_CALL_FUNC_X (x)

FLOAT
M_DECL_FUNC (__tgamma) (FLOAT x)
{
  int local_signgam;
  FLOAT y = M_CALL_FUNC (M_SUF (__ieee754_gamma)) (x, &local_signgam);

  if (__glibc_unlikely (!isfinite (y) || y == 0)
      && (isfinite (x) || (isinf (x) && x < 0)))
    {
      if (x == 0)
	/* Pole error: tgamma(x=0).  */
	__set_errno (ERANGE);
      else if (M_SUF (floor) (x) == x && x < 0)
	/* Domain error: tgamma(integer x<0).  */
	__set_errno (EDOM);
      else
	/* Overflow or underflow.  */
	__set_errno (ERANGE);
    }
  return local_signgam < 0 ? -y : y;
}
declare_mgen_alias (__tgamma, tgamma)

#endif /* __USE_WRAPPER_TEMPLATE.  */
