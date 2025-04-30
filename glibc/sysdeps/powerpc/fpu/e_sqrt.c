/* Double-precision floating point square root.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <math.h>
#include <math_private.h>
#include <fenv_libc.h>
#include <libm-alias-finite.h>
#include <math-use-builtins.h>

double
__ieee754_sqrt (double x)
{
#if USE_SQRT_BUILTIN
  return __builtin_sqrt (x);
#else
/* The method is based on a description in
   Computation of elementary functions on the IBM RISC System/6000 processor,
   P. W. Markstein, IBM J. Res. Develop, 34(1) 1990.
   Basically, it consists of two interleaved Newton-Raphson approximations,
   one to find the actual square root, and one to find its reciprocal
   without the expense of a division operation.   The tricky bit here
   is the use of the POWER/PowerPC multiply-add operation to get the
   required accuracy with high speed.

   The argument reduction works by a combination of table lookup to
   obtain the initial guesses, and some careful modification of the
   generated guesses (which mostly runs on the integer unit, while the
   Newton-Raphson is running on the FPU).  */

  extern const float __t_sqrt[1024];

  if (x > 0)
    {
      /* schedule the EXTRACT_WORDS to get separation between the store
	 and the load.  */
      ieee_double_shape_type ew_u;
      ieee_double_shape_type iw_u;
      ew_u.value = (x);
      if (x != INFINITY)
	{
	  /* Variables named starting with 's' exist in the
	     argument-reduced space, so that 2 > sx >= 0.5,
	     1.41... > sg >= 0.70.., 0.70.. >= sy > 0.35... .
	     Variables named ending with 'i' are integer versions of
	     floating-point values.  */
	  double sx;	/* The value of which we're trying to find the
			   square root.  */
	  double sg, g;	/* Guess of the square root of x.  */
	  double sd, d;	/* Difference between the square of the guess and x.  */
	  double sy;	/* Estimate of 1/2g (overestimated by 1ulp).  */
	  double sy2;	/* 2*sy */
	  double e;	/* Difference between y*g and 1/2 (se = e * fsy).  */
	  double shx;	/* == sx * fsg */
	  double fsg;	/* sg*fsg == g.  */
	  fenv_t fe;	/* Saved floating-point environment (stores rounding
			   mode and whether the inexact exception is
			   enabled).  */
	  uint32_t xi0, xi1, sxi, fsgi;
	  const float *t_sqrt;

	  fe = fegetenv_register ();
	  /* complete the EXTRACT_WORDS (xi0,xi1,x) operation.  */
	  xi0 = ew_u.parts.msw;
	  xi1 = ew_u.parts.lsw;
	  relax_fenv_state ();
	  sxi = (xi0 & 0x3fffffff) | 0x3fe00000;
	  /* schedule the INSERT_WORDS (sx, sxi, xi1) to get separation
	     between the store and the load.  */
	  iw_u.parts.msw = sxi;
	  iw_u.parts.lsw = xi1;
	  t_sqrt = __t_sqrt + (xi0 >> (52 - 32 - 8 - 1) & 0x3fe);
	  sg = t_sqrt[0];
	  sy = t_sqrt[1];
	  /* complete the INSERT_WORDS (sx, sxi, xi1) operation.  */
	  sx = iw_u.value;

	  /* Here we have three Newton-Raphson iterations each of a
	     division and a square root and the remainder of the
	     argument reduction, all interleaved.   */
	  sd = -__builtin_fma (sg, sg, -sx);
	  fsgi = (xi0 + 0x40000000) >> 1 & 0x7ff00000;
	  sy2 = sy + sy;
	  sg = __builtin_fma (sy, sd, sg);	/* 16-bit approximation to
						   sqrt(sx). */

	  /* schedule the INSERT_WORDS (fsg, fsgi, 0) to get separation
	     between the store and the load.  */
	  INSERT_WORDS (fsg, fsgi, 0);
	  iw_u.parts.msw = fsgi;
	  iw_u.parts.lsw = (0);
	  e = -__builtin_fma (sy, sg, -0x1.0000000000001p-1);
	  sd = -__builtin_fma (sg, sg, -sx);
	  if ((xi0 & 0x7ff00000) == 0)
	    goto denorm;
	  sy = __builtin_fma (e, sy2, sy);
	  sg = __builtin_fma (sy, sd, sg);	/* 32-bit approximation to
						   sqrt(sx).  */
	  sy2 = sy + sy;
	  /* complete the INSERT_WORDS (fsg, fsgi, 0) operation.  */
	  fsg = iw_u.value;
	  e = -__builtin_fma (sy, sg, -0x1.0000000000001p-1);
	  sd = -__builtin_fma (sg, sg, -sx);
	  sy = __builtin_fma (e, sy2, sy);
	  shx = sx * fsg;
	  sg = __builtin_fma (sy, sd, sg);	/* 64-bit approximation to
						   sqrt(sx), but perhaps
						   rounded incorrectly.  */
	  sy2 = sy + sy;
	  g = sg * fsg;
	  e = -__builtin_fma (sy, sg, -0x1.0000000000001p-1);
	  d = -__builtin_fma (g, sg, -shx);
	  sy = __builtin_fma (e, sy2, sy);
	  fesetenv_register (fe);
	  return __builtin_fma (sy, d, g);
	denorm:
	  /* For denormalised numbers, we normalise, calculate the
	     square root, and return an adjusted result.  */
	  fesetenv_register (fe);
	  return __ieee754_sqrt (x * 0x1p+108f) * 0x1p-54f;
	}
    }
  else if (x < 0)
    {
      /* For some reason, some PowerPC32 processors don't implement
	 FE_INVALID_SQRT.  */
# ifdef FE_INVALID_SQRT
      __feraiseexcept (FE_INVALID_SQRT);

      fenv_union_t u = { .fenv = fegetenv_register () };
      if ((u.l & FE_INVALID) == 0)
# endif
	__feraiseexcept (FE_INVALID);
      x = NAN;
    }
  return f_wash (x);
#endif /* USE_SQRT_BUILTIN  */
}

libm_alias_finite (__ieee754_sqrt, __sqrt)
