/* @(#)e_sinh.c 5.1 93/09/24 */
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

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: e_sinh.c,v 1.7 1995/05/10 20:46:13 jtc Exp $";
#endif

/* __ieee754_sinh(x)
 * Method :
 * mathematically sinh(x) if defined to be (exp(x)-exp(-x))/2
 *	1. Replace x by |x| (sinh(-x) = -sinh(x)).
 *	2.
 *						    E + E/(E+1)
 *	    0        <= x <= 22     :  sinh(x) := --------------, E=expm1(x)
 *							2
 *
 *	    22       <= x <= lnovft :  sinh(x) := exp(x)/2
 *	    lnovft   <= x <= ln2ovft:  sinh(x) := exp(x/2)/2 * exp(x/2)
 *	    ln2ovft  <  x	    :  sinh(x) := x*shuge (overflow)
 *
 * Special cases:
 *	sinh(x) is |x| if x is +INF, -INF, or NaN.
 *	only sinh(0)=0 is exact for finite x.
 */

#include <float.h>
#include <math.h>
#include <math-narrow-eval.h>
#include <math_private.h>
#include <math-underflow.h>
#include <libm-alias-finite.h>

static const double one = 1.0, shuge = 1.0e307;

double
__ieee754_sinh (double x)
{
  double t, w, h;
  int32_t ix, jx;
  uint32_t lx;

  /* High word of |x|. */
  GET_HIGH_WORD (jx, x);
  ix = jx & 0x7fffffff;

  /* x is INF or NaN */
  if (__glibc_unlikely (ix >= 0x7ff00000))
    return x + x;

  h = 0.5;
  if (jx < 0)
    h = -h;
  /* |x| in [0,22], return sign(x)*0.5*(E+E/(E+1))) */
  if (ix < 0x40360000)                  /* |x|<22 */
    {
      if (__glibc_unlikely (ix < 0x3e300000)) {            /* |x|<2**-28 */
	math_check_force_underflow (x);
	if (shuge + x > one)
	  return x;
	/* sinh(tiny) = tiny with inexact */
      }
      t = __expm1 (fabs (x));
      if (ix < 0x3ff00000)
	return h * (2.0 * t - t * t / (t + one));
      return h * (t + t / (t + one));
    }

  /* |x| in [22, log(maxdouble)] return 0.5*exp(|x|) */
  if (ix < 0x40862e42)
    return h * __ieee754_exp (fabs (x));

  /* |x| in [log(maxdouble), overflowthresold] */
  GET_LOW_WORD (lx, x);
  if (ix < 0x408633ce || ((ix == 0x408633ce) && (lx <= (uint32_t) 0x8fb9f87d)))
    {
      w = __ieee754_exp (0.5 * fabs (x));
      t = h * w;
      return t * w;
    }

  /* |x| > overflowthresold, sinh(x) overflow */
  return math_narrow_eval (x * shuge);
}
libm_alias_finite (__ieee754_sinh, __sinh)
