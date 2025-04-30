/* s_rintl.c -- long double version of s_rint.c.
 * Conversion to IEEE quad long double by Jakub Jelinek, jj@ultra.linux.cz.
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

#if defined (LIBM_SCCS) && ! defined (lint)
static char rcsid[] = "$NetBSD: $";
#endif

/*
 * rintl(x)
 * Return x rounded to integral value according to the prevailing
 * rounding mode.
 * Method:
 *	Using floating addition.
 * Exception:
 *	Inexact flag raised if x not equal to rintl(x).
 */

#define NO_MATH_REDIRECT
#include <math.h>
#include <math_private.h>
#include <libm-alias-ldouble.h>
#include <math-use-builtins.h>

_Float128
__rintl (_Float128 x)
{
#if USE_RINTL_BUILTIN
  return __builtin_rintl (x);
#else
  /* Use generic implementation.  */
  static const _Float128
    TWO112[2] = {
		 5.19229685853482762853049632922009600E+33L, /* 0x406F000000000000, 0 */
		 -5.19229685853482762853049632922009600E+33L  /* 0xC06F000000000000, 0 */
  };
  int64_t i0, j0, sx;
  uint64_t i1 __attribute__ ((unused));
  _Float128 w, t;
  GET_LDOUBLE_WORDS64 (i0, i1, x);
  sx = (((uint64_t) i0) >> 63);
  j0 = ((i0 >> 48) & 0x7fff) - 0x3fff;
  if (j0 < 112)
    {
      if (j0 < 0)
	{
	  w = TWO112[sx] + x;
	  t = w - TWO112[sx];
	  GET_LDOUBLE_MSW64 (i0, t);
	  SET_LDOUBLE_MSW64 (t, (i0 & 0x7fffffffffffffffLL) | (sx << 63));
	  return t;
	}
    }
  else
    {
      if (j0 == 0x4000)
	return x + x;		/* inf or NaN  */
      else
	return x;		/* x is integral  */
    }
  w = TWO112[sx] + x;
  return w - TWO112[sx];
#endif /* ! USE_RINTL_BUILTIN  */
}
libm_alias_ldouble (__rint, rint)
