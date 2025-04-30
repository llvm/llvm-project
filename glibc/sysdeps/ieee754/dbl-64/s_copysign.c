/* @(#)s_copysign.c 5.1 93/09/24 */
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
static char rcsid[] = "$NetBSD: s_copysign.c,v 1.8 1995/05/10 20:46:57 jtc Exp $";
#endif

/*
 * copysign(double x, double y)
 * copysign(x,y) returns a value with the magnitude of x and
 * with the sign bit of y.
 */

#define NO_MATH_REDIRECT
#include <math.h>
#include <libm-alias-double.h>

double
__copysign (double x, double y)
{
  return __builtin_copysign (x, y);
}
libm_alias_double (__copysign, copysign)
