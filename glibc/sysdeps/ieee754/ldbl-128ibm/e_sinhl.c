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


/* __ieee754_sinh(x)
 * Method :
 * mathematically sinh(x) if defined to be (exp(x)-exp(-x))/2
 *	1. Replace x by |x| (sinh(-x) = -sinh(x)).
 *	2.
 *						    E + E/(E+1)
 *	    0        <= x <= 40     :  sinh(x) := --------------, E=expm1(x)
 *							2
 *
 *	    40       <= x <= lnovft :  sinh(x) := exp(x)/2
 *	    lnovft   <= x <= ln2ovft:  sinh(x) := exp(x/2)/2 * exp(x/2)
 *	    ln2ovft  <  x	    :  sinh(x) := x*shuge (overflow)
 *
 * Special cases:
 *	sinh(x) is |x| if x is +INF, -INF, or NaN.
 *	only sinh(0)=0 is exact for finite x.
 */

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <libm-alias-finite.h>

static const long double one = 1.0, shuge = 1.0e307;

long double
__ieee754_sinhl(long double x)
{
	long double t,w,h;
	int64_t ix,jx;
	double xhi;

    /* High word of |x|. */
	xhi = ldbl_high (x);
	EXTRACT_WORDS64 (jx, xhi);
	ix = jx&0x7fffffffffffffffLL;

    /* x is INF or NaN */
	if(ix>=0x7ff0000000000000LL) return x+x;

	h = 0.5;
	if (jx<0) h = -h;
    /* |x| in [0,40], return sign(x)*0.5*(E+E/(E+1))) */
	if (ix < 0x4044000000000000LL) {	/* |x|<40 */
	    if (ix<0x3c90000000000000LL) {	/* |x|<2**-54 */
		math_check_force_underflow (x);
		if(shuge+x>one) return x;/* sinhl(tiny) = tiny with inexact */
	    }
	    t = __expm1l(fabsl(x));
	    if(ix<0x3ff0000000000000LL) return h*(2.0*t-t*t/(t+one));
	    w = t/(t+one);
	    return h*(t+w);
	}

    /* |x| in [40, log(maxdouble)] return 0.5*exp(|x|) */
	if (ix < 0x40862e42fefa39efLL)  return h*__ieee754_expl(fabsl(x));

    /* |x| in [log(maxdouble), overflowthresold] */
	if (ix <= 0x408633ce8fb9f87eLL) {
	    w = __ieee754_expl(0.5*fabsl(x));
	    t = h*w;
	    return t*w;
	}

    /* |x| > overflowthresold, sinh(x) overflow */
	return x*shuge;
}
libm_alias_finite (__ieee754_sinhl, __sinhl)
