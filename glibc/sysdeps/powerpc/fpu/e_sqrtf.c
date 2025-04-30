/* Single-precision floating point square root.
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

float
__ieee754_sqrtf (float x)
{
#if USE_SQRTF_BUILTIN
  return __builtin_sqrtf (x);
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
      if (x != INFINITY)
	{
	  /* Variables named starting with 's' exist in the
	     argument-reduced space, so that 2 > sx >= 0.5,
	     1.41... > sg >= 0.70.., 0.70.. >= sy > 0.35... .
	     Variables named ending with 'i' are integer versions of
	     floating-point values.  */
	  float sx;		/* The value of which we're trying to find the square
				   root.  */
	  float sg, g;		/* Guess of the square root of x.  */
	  float sd, d;		/* Difference between the square of the guess and x.  */
	  float sy;		/* Estimate of 1/2g (overestimated by 1ulp).  */
	  float sy2;		/* 2*sy */
	  float e;		/* Difference between y*g and 1/2 (note that e==se).  */
	  float shx;		/* == sx * fsg */
	  float fsg;		/* sg*fsg == g.  */
	  fenv_t fe;		/* Saved floating-point environment (stores rounding
				   mode and whether the inexact exception is
				   enabled).  */
	  uint32_t xi, sxi, fsgi;
	  const float *t_sqrt;

	  GET_FLOAT_WORD (xi, x);
	  fe = fegetenv_register ();
	  relax_fenv_state ();
	  sxi = (xi & 0x3fffffff) | 0x3f000000;
	  SET_FLOAT_WORD (sx, sxi);
	  t_sqrt = __t_sqrt + (xi >> (23 - 8 - 1) & 0x3fe);
	  sg = t_sqrt[0];
	  sy = t_sqrt[1];

	  /* Here we have three Newton-Raphson iterations each of a
	     division and a square root and the remainder of the
	     argument reduction, all interleaved.   */
	  sd = -__builtin_fmaf (sg, sg, -sx);
	  fsgi = (xi + 0x40000000) >> 1 & 0x7f800000;
	  sy2 = sy + sy;
	  sg = __builtin_fmaf (sy, sd, sg);	/* 16-bit approximation to
						   sqrt(sx). */
	  e = -__builtin_fmaf (sy, sg, -0x1.0000020365653p-1);
	  SET_FLOAT_WORD (fsg, fsgi);
	  sd = -__builtin_fmaf (sg, sg, -sx);
	  sy = __builtin_fmaf (e, sy2, sy);
	  if ((xi & 0x7f800000) == 0)
	    goto denorm;
	  shx = sx * fsg;
	  sg = __builtin_fmaf (sy, sd, sg);	/* 32-bit approximation to
						   sqrt(sx), but perhaps
						   rounded incorrectly.  */
	  sy2 = sy + sy;
	  g = sg * fsg;
	  e = -__builtin_fmaf (sy, sg, -0x1.0000020365653p-1);
	  d = -__builtin_fmaf (g, sg, -shx);
	  sy = __builtin_fmaf (e, sy2, sy);
	  fesetenv_register (fe);
	  return __builtin_fmaf (sy, d, g);
	denorm:
	  /* For denormalised numbers, we normalise, calculate the
	     square root, and return an adjusted result.  */
	  fesetenv_register (fe);
	  return __ieee754_sqrtf (x * 0x1p+48) * 0x1p-24;
	}
    }
  else if (x < 0)
    {
      /* For some reason, some PowerPC32 processors don't implement
	 FE_INVALID_SQRT.  */
# ifdef FE_INVALID_SQRT
      feraiseexcept (FE_INVALID_SQRT);

      fenv_union_t u = { .fenv = fegetenv_register () };
      if ((u.l & FE_INVALID) == 0)
# endif
	feraiseexcept (FE_INVALID);
      x = NAN;
    }
  return f_washf (x);
#endif /* USE_SQRTF_BUILTIN  */
}
libm_alias_finite (__ieee754_sqrtf, __sqrtf)
