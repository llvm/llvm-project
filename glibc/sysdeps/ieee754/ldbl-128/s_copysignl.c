/* s_copysignl.c -- long double version of s_copysign.c.
 * Conversion to long double by Jakub Jelinek, jj@ultra.linux.cz.
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
 * copysignl(long double x, long double y)
 * copysignl(x,y) returns a value with the magnitude of x and
 * with the sign bit of y.
 */

#define NO_MATH_REDIRECT
#include <math.h>
#include <math_private.h>
#include <libm-alias-ldouble.h>
#include <math-use-builtins.h>

_Float128
__copysignl (_Float128 x, _Float128 y)
{
#if USE_COPYSIGNL_BUILTIN
  return __builtin_copysignl (x, y);
#else
  /* Use generic implementation.  */
  uint64_t hx, hy;
  GET_LDOUBLE_MSW64 (hx, x);
  GET_LDOUBLE_MSW64 (hy, y);
  SET_LDOUBLE_MSW64 (x, (hx & 0x7fffffffffffffffULL)
		     | (hy & 0x8000000000000000ULL));
  return x;
#endif /* ! USE_COPYSIGNL_BUILTIN  */
}
libm_alias_ldouble (__copysign, copysign)
