/*
 * IBM Accurate Mathematical Library
 * written by International Business Machines Corp.
 * Copyright (C) 2001-2021 Free Software Foundation, Inc.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, see <https://www.gnu.org/licenses/>.
 */
/*********************************************************************/
/* MODULE_NAME: uroot.c                                              */
/*                                                                   */
/* FUNCTION:    usqrt                                                */
/*                                                                   */
/* FILES NEEDED: dla.h endian.h mydefs.h                             */
/*               uroot.tbl                                           */
/*                                                                   */
/* An ultimate sqrt routine. Given an IEEE double machine number x   */
/* it computes the correctly rounded (to nearest) value of square    */
/* root of x.                                                        */
/* Assumption: Machine arithmetic operations are performed in        */
/* round to nearest mode of IEEE 754 standard.                       */
/*                                                                   */
/*********************************************************************/

#include "endian.h"
#include "mydefs.h"
#include <dla.h>
#include "root.tbl"
#include <math-barriers.h>
#include <math_private.h>
#include <fenv_private.h>
#include <libm-alias-finite.h>
#include <math-use-builtins.h>

/*********************************************************************/
/* An ultimate sqrt routine. Given an IEEE double machine number x   */
/* it computes the correctly rounded (to nearest) value of square    */
/* root of x.                                                        */
/*********************************************************************/
double
__ieee754_sqrt (double x)
{
#if USE_SQRT_BUILTIN
  return __builtin_sqrt (x);
#else
  /* Use generic implementation.  */
  static const double
    rt0 = 9.99999999859990725855365213134618E-01,
    rt1 = 4.99999999495955425917856814202739E-01,
    rt2 = 3.75017500867345182581453026130850E-01,
    rt3 = 3.12523626554518656309172508769531E-01;
  static const double big = 134217728.0;
  double y, t, del, res, res1, hy, z, zz, s;
  mynumber a, c = { { 0, 0 } };
  int4 k;

  a.x = x;
  k = a.i[HIGH_HALF];
  a.i[HIGH_HALF] = (k & 0x001fffff) | 0x3fe00000;
  t = inroot[(k & 0x001fffff) >> 14];
  s = a.x;
  /*----------------- 2^-1022  <= | x |< 2^1024  -----------------*/
  if (k > 0x000fffff && k < 0x7ff00000)
    {
      int rm = __fegetround ();
      fenv_t env;
      libc_feholdexcept_setround (&env, FE_TONEAREST);
      double ret;
      y = 1.0 - t * (t * s);
      t = t * (rt0 + y * (rt1 + y * (rt2 + y * rt3)));
      c.i[HIGH_HALF] = 0x20000000 + ((k & 0x7fe00000) >> 1);
      y = t * s;
      hy = (y + big) - big;
      del = 0.5 * t * ((s - hy * hy) - (y - hy) * (y + hy));
      res = y + del;
      if (res == (res + 1.002 * ((y - res) + del)))
	ret = res * c.x;
      else
	{
	  res1 = res + 1.5 * ((y - res) + del);
	  EMULV (res, res1, z, zz); /* (z+zz)=res*res1 */
	  res = ((((z - s) + zz) < 0) ? max (res, res1) :
					min (res, res1));
	  ret = res * c.x;
	}
      math_force_eval (ret);
      libc_fesetenv (&env);
      double dret = x / ret;
      if (dret != ret)
	{
	  double force_inexact = 1.0 / 3.0;
	  math_force_eval (force_inexact);
	  /* The square root is inexact, ret is the round-to-nearest
	     value which may need adjusting for other rounding
	     modes.  */
	  switch (rm)
	    {
#ifdef FE_UPWARD
	    case FE_UPWARD:
	      if (dret > ret)
		ret = (res + 0x1p-1022) * c.x;
	      break;
#endif

#ifdef FE_DOWNWARD
	    case FE_DOWNWARD:
#endif
#ifdef FE_TOWARDZERO
	    case FE_TOWARDZERO:
#endif
#if defined FE_DOWNWARD || defined FE_TOWARDZERO
	      if (dret < ret)
		ret = (res - 0x1p-1022) * c.x;
	      break;
#endif

	    default:
	      break;
	    }
	}
      /* Otherwise (x / ret == ret), either the square root was exact or
         the division was inexact.  */
      return ret;
    }
  else
    {
      if ((k & 0x7ff00000) == 0x7ff00000)
	return x * x + x; /* sqrt(NaN)=NaN, sqrt(+inf)=+inf, sqrt(-inf)=sNaN */
      if (x == 0)
	return x;       /* sqrt(+0)=+0, sqrt(-0)=-0 */
      if (k < 0)
	return (x - x) / (x - x); /* sqrt(-ve)=sNaN */
      return 0x1p-256 * __ieee754_sqrt (x * 0x1p512);
    }
#endif /* ! USE_SQRT_BUILTIN  */
}
#ifndef __ieee754_sqrt
libm_alias_finite (__ieee754_sqrt, __sqrt)
#endif
