/* s_fabsf.c -- float version of s_fabs.c.
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

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: s_fabsf.c,v 1.4 1995/05/10 20:47:15 jtc Exp $";
#endif

/*
 * fabsf(x) returns the absolute value of x.
 */

#include <math.h>
#include <libm-alias-float.h>

float __fabsf(float x)
{
  return __builtin_fabsf (x);
}
libm_alias_float (__fabs, fabs)
