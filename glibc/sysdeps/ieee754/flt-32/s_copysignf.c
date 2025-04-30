/* s_copysignf.c -- float version of s_copysign.c.
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

#if defined (LIBM_SCCS) && ! defined (lint)
static char rcsid[] = "$NetBSD: s_copysignf.c,v 1.4 1995/05/10 20:46:59 jtc Exp $";
#endif

/*
 * copysignf(float x, float y)
 * copysignf(x,y) returns a value with the magnitude of x and
 * with the sign bit of y.
 */

#define NO_MATH_REDIRECT
#include <math.h>
#include <libm-alias-float.h>

float
__copysignf (float x, float y)
{
  return __builtin_copysignf (x, y);
}
libm_alias_float (__copysign, copysign)
