/* @(#)e_hypotl.c 5.1 93/09/24 */
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

/* __ieee754_hypotl(x,y)
 *
 * Method :
 *	If (assume round-to-nearest) z=x*x+y*y
 *	has error less than sqrtl(2)/2 ulp, than
 *	sqrtl(z) has error less than 1 ulp (exercise).
 *
 *	So, compute sqrtl(x*x+y*y) with some care as
 *	follows to get the error below 1 ulp:
 *
 *	Assume x>y>0;
 *	(if possible, set rounding to round-to-nearest)
 *	1. if x > 2y  use
 *		x1*x1+(y*y+(x2*(x+x1))) for x*x+y*y
 *	where x1 = x with lower 53 bits cleared, x2 = x-x1; else
 *	2. if x <= 2y use
 *		t1*y1+((x-y)*(x-y)+(t1*y2+t2*y))
 *	where t1 = 2x with lower 53 bits cleared, t2 = 2x-t1,
 *	y1= y with lower 53 bits chopped, y2 = y-y1.
 *
 *	NOTE: scaling may be necessary if some argument is too
 *	      large or too tiny
 *
 * Special cases:
 *	hypotl(x,y) is INF if x or y is +INF or -INF; else
 *	hypotl(x,y) is NAN if x or y is NAN.
 *
 * Accuracy:
 *	hypotl(x,y) returns sqrtl(x^2+y^2) with error less
 *	than 1 ulps (units in the last place)
 */

#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <libm-alias-finite.h>

long double
__ieee754_hypotl(long double x, long double y)
{
	long double a,b,a1,a2,b1,b2,w,kld;
	int64_t j,k,ha,hb;
	double xhi, yhi, hi, lo;

	xhi = ldbl_high (x);
	EXTRACT_WORDS64 (ha, xhi);
	yhi = ldbl_high (y);
	EXTRACT_WORDS64 (hb, yhi);
	ha &= 0x7fffffffffffffffLL;
	hb &= 0x7fffffffffffffffLL;
	if(hb > ha) {a=y;b=x;j=ha; ha=hb;hb=j;} else {a=x;b=y;}
	a = fabsl(a);	/* a <- |a| */
	b = fabsl(b);	/* b <- |b| */
	if((ha-hb)>0x0780000000000000LL) {return a+b;} /* x/y > 2**120 */
	k=0;
	kld = 1.0L;
	if(ha > 0x5f30000000000000LL) {	/* a>2**500 */
	   if(ha >= 0x7ff0000000000000LL) {	/* Inf or NaN */
	       w = a+b;			/* for sNaN */
	       if (issignaling (a) || issignaling (b))
		 return w;
	       if(ha == 0x7ff0000000000000LL)
		 w = a;
	       if(hb == 0x7ff0000000000000LL)
		 w = b;
	       return w;
	   }
	   /* scale a and b by 2**-600 */
	   a *= 0x1p-600L;
	   b *= 0x1p-600L;
	   k = 600;
	   kld = 0x1p+600L;
	}
	else if(hb < 0x23d0000000000000LL) {	/* b < 2**-450 */
	    if(hb <= 0x000fffffffffffffLL) {	/* subnormal b or 0 */
		if(hb==0) return a;
		a *= 0x1p+1022L;
		b *= 0x1p+1022L;
		k = -1022;
		kld = 0x1p-1022L;
	    } else {		/* scale a and b by 2^600 */
		a *= 0x1p+600L;
		b *= 0x1p+600L;
		k = -600;
		kld = 0x1p-600L;
	    }
	}
    /* medium size a and b */
	w = a-b;
	if (w>b) {
	    ldbl_unpack (a, &hi, &lo);
	    a1 = hi;
	    a2 = lo;
	    /* a*a + b*b
	       = (a1+a2)*a + b*b
	       = a1*a + a2*a + b*b
	       = a1*(a1+a2) + a2*a + b*b
	       = a1*a1 + a1*a2 + a2*a + b*b
	       = a1*a1 + a2*(a+a1) + b*b  */
	    w  = sqrtl(a1*a1-(b*(-b)-a2*(a+a1)));
	} else {
	    a  = a+a;
	    ldbl_unpack (b, &hi, &lo);
	    b1 = hi;
	    b2 = lo;
	    ldbl_unpack (a, &hi, &lo);
	    a1 = hi;
	    a2 = lo;
	    /* a*a + b*b
	       = a*a + (a-b)*(a-b) - (a-b)*(a-b) + b*b
	       = a*a + w*w  - (a*a - 2*a*b + b*b) + b*b
	       = w*w + 2*a*b
	       = w*w + (a1+a2)*b
	       = w*w + a1*b + a2*b
	       = w*w + a1*(b1+b2) + a2*b
	       = w*w + a1*b1 + a1*b2 + a2*b  */
	    w  = sqrtl(a1*b1-(w*(-w)-(a1*b2+a2*b)));
	}
	if(k!=0)
	    {
		w *= kld;
		math_check_force_underflow_nonneg (w);
		return w;
	    }
	else
	    return w;
}
libm_alias_finite (__ieee754_hypotl, __hypotl)
