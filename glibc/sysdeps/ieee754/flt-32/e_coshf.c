/* e_coshf.c -- float version of e_cosh.c.
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 * Optimizations by Ulrich Drepper <drepper@gmail.com>, 2011
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

#include <math.h>
#include <math-narrow-eval.h>
#include <math_private.h>
#include <libm-alias-finite.h>

static const float huge = 1.0e30;
static const float one = 1.0, half=0.5;

float
__ieee754_coshf (float x)
{
	float t,w;
	int32_t ix;

	GET_FLOAT_WORD(ix,x);
	ix &= 0x7fffffff;

    /* |x| in [0,22] */
	if (ix < 0x41b00000) {
	    /* |x| in [0,0.5*ln2], return 1+expm1(|x|)^2/(2*exp(|x|)) */
		if(ix<0x3eb17218) {
		    if (ix<0x24000000) return one;	/* cosh(tiny) = 1 */
		    t = __expm1f(fabsf(x));
		    w = one+t;
		    return one+(t*t)/(w+w);
		}

	    /* |x| in [0.5*ln2,22], return (exp(|x|)+1/exp(|x|)/2; */
		t = __ieee754_expf(fabsf(x));
		return half*t+half/t;
	}

    /* |x| in [22, log(maxdouble)] return half*exp(|x|) */
	if (ix < 0x42b17180)  return half*__ieee754_expf(fabsf(x));

    /* |x| in [log(maxdouble), overflowthresold] */
	if (ix<=0x42b2d4fc) {
	    w = __ieee754_expf(half*fabsf(x));
	    t = half*w;
	    return t*w;
	}

    /* x is INF or NaN */
	if(ix>=0x7f800000) return x*x;

    /* |x| > overflowthresold, cosh(x) overflow */
	return math_narrow_eval (huge*huge);
}
libm_alias_finite (__ieee754_coshf, __coshf)
