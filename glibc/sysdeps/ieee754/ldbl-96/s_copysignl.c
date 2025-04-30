/* s_copysignl.c -- long double version of s_copysign.c.
 * Conversion to long double by Ulrich Drepper,
 * Cygnus Support, drepper@cygnus.com.
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
#include <math_private.h>
#include <libm-alias-ldouble.h>

long double __copysignl(long double x, long double y)
{
	uint32_t es1,es2;
	GET_LDOUBLE_EXP(es1,x);
	GET_LDOUBLE_EXP(es2,y);
	SET_LDOUBLE_EXP(x,(es1&0x7fff)|(es2&0x8000));
        return x;
}
libm_alias_ldouble (__copysign, copysign)
