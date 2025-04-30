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

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: $";
#endif

/*
 * copysignl(long double x, long double y)
 * copysignl(x,y) returns a value with the magnitude of x and
 * with the sign bit of y.
 */

#define NO_MATH_REDIRECT
#include <math.h>
#include <math_ldbl_opt.h>

long double __copysignl(long double x, long double y)
{
  if (signbit (x) != signbit (y))
    x = -x;
  return x;
}

#if IS_IN (libm)
long_double_symbol (libm, __copysignl, copysignl);
#else
long_double_symbol (libc, __copysignl, copysignl);
#endif
