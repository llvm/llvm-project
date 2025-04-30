/* s_rintf.c -- float version of s_rint.c.
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

#define NO_MATH_REDIRECT
#include <math.h>
#include <math_private.h>
#include <libm-alias-float.h>
#include <math-use-builtins.h>

float
__rintf (float x)
{
#if USE_RINTF_BUILTIN
  return __builtin_rintf (x);
#else
  /* Use generic implementation.  */
  static const float
    TWO23[2] = {
		8.3886080000e+06, /* 0x4b000000 */
		-8.3886080000e+06, /* 0xcb000000 */
  };
  int32_t i0, j0, sx;
  float w, t;
  GET_FLOAT_WORD (i0, x);
  sx = (i0 >> 31) & 1;
  j0 = ((i0 >> 23) & 0xff) - 0x7f;
  if (j0 < 23)
    {
      if(j0 < 0)
	{
	  w = TWO23[sx] + x;
	  t =  w - TWO23[sx];
	  GET_FLOAT_WORD (i0, t);
	  SET_FLOAT_WORD (t, (i0 & 0x7fffffff) | (sx << 31));
	  return t;
	}
    }
  else
    {
      if (j0 == 0x80)
	return x + x;		/* inf or NaN  */
      else
	return x;		/* x is integral  */
    }
  w = TWO23[sx] + x;
  return w - TWO23[sx];
#endif /* ! USE_RINTF_BUILTIN  */
}
#ifndef __rintf
libm_alias_float (__rint, rint)
#endif
