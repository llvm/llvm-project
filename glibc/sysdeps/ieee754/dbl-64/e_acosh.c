/* Optimized for 64-bit by Ulrich Drepper <drepper@gmail.com>, 2012 */
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

/* __ieee754_acosh(x)
 * Method :
 *	Based on
 *		acosh(x) = log [ x + sqrt(x*x-1) ]
 *	we have
 *		acosh(x) := log(x)+ln2,	if x is large; else
 *		acosh(x) := log(2x-1/(sqrt(x*x-1)+x)) if x>2; else
 *		acosh(x) := log1p(t+sqrt(2.0*t+t*t)); where t=x-1.
 *
 * Special cases:
 *	acosh(x) is NaN with signal if x<1.
 *	acosh(NaN) is NaN without signal.
 */

#include <math.h>
#include <math_private.h>
#include <libm-alias-finite.h>

static const double
one	= 1.0,
ln2	= 6.93147180559945286227e-01;  /* 0x3FE62E42, 0xFEFA39EF */

double
__ieee754_acosh (double x)
{
  int64_t hx;
  EXTRACT_WORDS64 (hx, x);

  if (hx > INT64_C (0x4000000000000000))
    {
      if (__glibc_unlikely (hx >= INT64_C (0x41b0000000000000)))
	{
	  /* x > 2**28 */
	  if (hx >= INT64_C (0x7ff0000000000000))
	    /* x is inf of NaN */
	    return x + x;
	  else
	    return __ieee754_log (x) + ln2;/* acosh(huge)=log(2x) */
	}

      /* 2**28 > x > 2 */
      double t = x * x;
      return __ieee754_log (2.0 * x - one / (x + sqrt (t - one)));
    }
  else if (__glibc_likely (hx > INT64_C (0x3ff0000000000000)))
    {
      /* 1<x<2 */
      double t = x - one;
      return __log1p (t + sqrt (2.0 * t + t * t));
    }
  else if (__glibc_likely (hx == INT64_C (0x3ff0000000000000)))
    return 0.0;				/* acosh(1) = 0 */
  else					/* x < 1 */
    return (x - x) / (x - x);
}
libm_alias_finite (__ieee754_acosh, __acosh)
