/* e_acoshf.c -- float version of e_acosh.c.
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
static char rcsid[] = "$NetBSD: e_acoshf.c,v 1.5 1995/05/12 04:57:20 jtc Exp $";
#endif

#include <math.h>
#include <math_private.h>
#include <libm-alias-finite.h>

static const float
one	= 1.0,
ln2	= 6.9314718246e-01;  /* 0x3f317218 */

float __ieee754_acoshf(float x)
{
	float t;
	int32_t hx;
	GET_FLOAT_WORD(hx,x);
	if(hx<0x3f800000) {		/* x < 1 */
	    return (x-x)/(x-x);
	} else if(hx >=0x4d800000) {	/* x > 2**28 */
	    if(hx >=0x7f800000) {	/* x is inf of NaN */
		return x+x;
	    } else
		return __ieee754_logf(x)+ln2;	/* acosh(huge)=log(2x) */
	} else if (hx==0x3f800000) {
	    return 0.0;			/* acosh(1) = 0 */
	} else if (hx > 0x40000000) {	/* 2**28 > x > 2 */
	    t=x*x;
	    return __ieee754_logf((float)2.0*x-one/(x+sqrtf(t-one)));
	} else {			/* 1<x<2 */
	    t = x-one;
	    return __log1pf(t+sqrtf((float)2.0*t+t*t));
	}
}
libm_alias_finite (__ieee754_acoshf, __acoshf)
