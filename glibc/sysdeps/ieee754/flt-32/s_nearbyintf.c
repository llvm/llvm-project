/* s_rintf.c -- float version of s_rint.c.
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */
/* Adapted for use as nearbyint by Ulrich Drepper <drepper@cygnus.com>.  */

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


#include <fenv.h>
#include <math.h>
#include <math-barriers.h>
#include <math_private.h>
#include <fenv_private.h>
#include <libm-alias-float.h>
#include <math-use-builtins.h>

float
__nearbyintf (float x)
{
#if USE_NEARBYINTF_BUILTIN
  return __builtin_nearbyintf (x);
#else
  /* Use generic implementation.  */
  static const float
    TWO23[2] = {
		8.3886080000e+06, /* 0x4b000000 */
		-8.3886080000e+06, /* 0xcb000000 */
  };
  fenv_t env;
  int32_t i0, j0, sx;
  float w, t;
  GET_FLOAT_WORD (i0, x);
  sx = (i0 >> 31) & 1;
  j0 = ((i0 >> 23) & 0xff) - 0x7f;
  if (j0 < 23)
    {
      if (j0 < 0)
	{
	  libc_feholdexceptf (&env);
	  w = TWO23[sx] + math_opt_barrier (x);
	  t =  w - TWO23[sx];
	  math_force_eval (t);
	  libc_fesetenvf (&env);
	  GET_FLOAT_WORD (i0, t);
	  SET_FLOAT_WORD (t, (i0 & 0x7fffffff) | (sx << 31));
	  return t;
	}
    }
  else
    {
      if (__glibc_unlikely (j0 == 0x80))
	return x + x;		/* inf or NaN  */
      else
	return x;		/* x is integral  */
  }
  libc_feholdexceptf (&env);
  w = TWO23[sx] + math_opt_barrier (x);
  t = w - TWO23[sx];
  math_force_eval (t);
  libc_fesetenvf (&env);
  return t;
#endif /* ! USE_NEARBYINT_BUILTIN  */
}
libm_alias_float (__nearbyint, nearbyint)
