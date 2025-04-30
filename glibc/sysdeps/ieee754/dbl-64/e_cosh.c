/* Optimized by Ulrich Drepper <drepper@gmail.com>, 2011 */
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

/* __ieee754_cosh(x)
 * Method :
 * mathematically cosh(x) if defined to be (exp(x)+exp(-x))/2
 *	1. Replace x by |x| (cosh(x) = cosh(-x)).
 *	2.
 *							[ exp(x) - 1 ]^2
 *	    0        <= x <= ln2/2  :  cosh(x) := 1 + -------------------
 *							   2*exp(x)
 *
 *						  exp(x) +  1/exp(x)
 *	    ln2/2    <= x <= 22     :  cosh(x) := -------------------
 *							  2
 *	    22       <= x <= lnovft :  cosh(x) := exp(x)/2
 *	    lnovft   <= x <= ln2ovft:  cosh(x) := exp(x/2)/2 * exp(x/2)
 *	    ln2ovft  <  x	    :  cosh(x) := huge*huge (overflow)
 *
 * Special cases:
 *	cosh(x) is |x| if x is +INF, -INF, or NaN.
 *	only cosh(0)=1 is exact for finite x.
 */

#include <math.h>
#include <math_private.h>
#include <math-narrow-eval.h>
#include <libm-alias-finite.h>

static const double one = 1.0, half=0.5, huge = 1.0e300;

double
__ieee754_cosh (double x)
{
	double t,w;
	int32_t ix;

    /* High word of |x|. */
	GET_HIGH_WORD(ix,x);
	ix &= 0x7fffffff;

    /* |x| in [0,22] */
	if (ix < 0x40360000) {
	    /* |x| in [0,0.5*ln2], return 1+expm1(|x|)^2/(2*exp(|x|)) */
		if(ix<0x3fd62e43) {
		    if (ix<0x3c800000)			/* cosh(tiny) = 1 */
		      return one;
		    t = __expm1(fabs(x));
		    w = one+t;
		    return one+(t*t)/(w+w);
		}

	    /* |x| in [0.5*ln2,22], return (exp(|x|)+1/exp(|x|)/2; */
		t = __ieee754_exp(fabs(x));
		return half*t+half/t;
	}

    /* |x| in [22, log(maxdouble)] return half*exp(|x|) */
	if (ix < 0x40862e42)  return half*__ieee754_exp(fabs(x));

    /* |x| in [log(maxdouble), overflowthresold] */
	int64_t fix;
	EXTRACT_WORDS64(fix, x);
	fix &= UINT64_C(0x7fffffffffffffff);
	if (fix <= UINT64_C(0x408633ce8fb9f87d)) {
	    w = __ieee754_exp(half*fabs(x));
	    t = half*w;
	    return t*w;
	}

    /* x is INF or NaN */
	if(ix>=0x7ff00000) return x*x;

    /* |x| > overflowthresold, cosh(x) overflow */
	return math_narrow_eval (huge * huge);
}
libm_alias_finite (__ieee754_cosh, __cosh)
