/* e_hypotl.c -- long double version of e_hypot.c.
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

/* __ieee754_hypotl(x,y)
 *
 * Method :
 *	If (assume round-to-nearest) z=x*x+y*y
 *	has error less than sqrt(2)/2 ulp, than
 *	sqrt(z) has error less than 1 ulp (exercise).
 *
 *	So, compute sqrt(x*x+y*y) with some care as
 *	follows to get the error below 1 ulp:
 *
 *	Assume x>y>0;
 *	(if possible, set rounding to round-to-nearest)
 *	1. if x > 2y  use
 *		x1*x1+(y*y+(x2*(x+x1))) for x*x+y*y
 *	where x1 = x with lower 32 bits cleared, x2 = x-x1; else
 *	2. if x <= 2y use
 *		t1*y1+((x-y)*(x-y)+(t1*y2+t2*y))
 *	where t1 = 2x with lower 32 bits cleared, t2 = 2x-t1,
 *	y1= y with lower 32 bits chopped, y2 = y-y1.
 *
 *	NOTE: scaling may be necessary if some argument is too
 *	      large or too tiny
 *
 * Special cases:
 *	hypot(x,y) is INF if x or y is +INF or -INF; else
 *	hypot(x,y) is NAN if x or y is NAN.
 *
 * Accuracy:
 *	hypot(x,y) returns sqrt(x^2+y^2) with error less
 *	than 1 ulps (units in the last place)
 */

#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <libm-alias-finite.h>

long double __ieee754_hypotl(long double x, long double y)
{
	long double a,b,t1,t2,y1,y2,w;
	uint32_t j,k,ea,eb;

	GET_LDOUBLE_EXP(ea,x);
	ea &= 0x7fff;
	GET_LDOUBLE_EXP(eb,y);
	eb &= 0x7fff;
	if(eb > ea) {a=y;b=x;j=ea; ea=eb;eb=j;} else {a=x;b=y;}
	SET_LDOUBLE_EXP(a,ea);	/* a <- |a| */
	SET_LDOUBLE_EXP(b,eb);	/* b <- |b| */
	if((ea-eb)>0x46) {return a+b;} /* x/y > 2**70 */
	k=0;
	if(__builtin_expect(ea > 0x5f3f,0)) {	/* a>2**8000 */
	   if(ea == 0x7fff) {	/* Inf or NaN */
	       uint32_t exp __attribute__ ((unused));
	       uint32_t high,low;
	       w = a+b;			/* for sNaN */
	       if (issignaling (a) || issignaling (b))
		 return w;
	       GET_LDOUBLE_WORDS(exp,high,low,a);
	       if(((high&0x7fffffff)|low)==0) w = a;
	       GET_LDOUBLE_WORDS(exp,high,low,b);
	       if(((eb^0x7fff)|(high&0x7fffffff)|low)==0) w = b;
	       return w;
	   }
	   /* scale a and b by 2**-9600 */
	   ea -= 0x2580; eb -= 0x2580;	k += 9600;
	   SET_LDOUBLE_EXP(a,ea);
	   SET_LDOUBLE_EXP(b,eb);
	}
	if(__builtin_expect(eb < 0x20bf, 0)) {	/* b < 2**-8000 */
	    if(eb == 0) {	/* subnormal b or 0 */
		uint32_t exp __attribute__ ((unused));
		uint32_t high,low;
		GET_LDOUBLE_WORDS(exp,high,low,b);
		if((high|low)==0) return a;
		SET_LDOUBLE_WORDS(t1, 0x7ffd, 0x80000000, 0); /* t1=2^16382 */
		b *= t1;
		a *= t1;
		k -= 16382;
		GET_LDOUBLE_EXP (ea, a);
		GET_LDOUBLE_EXP (eb, b);
		if (eb > ea)
		  {
		    t1 = a;
		    a = b;
		    b = t1;
		    j = ea;
		    ea = eb;
		    eb = j;
		  }
	    } else {		/* scale a and b by 2^9600 */
		ea += 0x2580;	/* a *= 2^9600 */
		eb += 0x2580;	/* b *= 2^9600 */
		k -= 9600;
		SET_LDOUBLE_EXP(a,ea);
		SET_LDOUBLE_EXP(b,eb);
	    }
	}
    /* medium size a and b */
	w = a-b;
	if (w>b) {
	    uint32_t high;
	    GET_LDOUBLE_MSW(high,a);
	    SET_LDOUBLE_WORDS(t1,ea,high,0);
	    t2 = a-t1;
	    w  = sqrtl(t1*t1-(b*(-b)-t2*(a+t1)));
	} else {
	    uint32_t high;
	    GET_LDOUBLE_MSW(high,b);
	    a  = a+a;
	    SET_LDOUBLE_WORDS(y1,eb,high,0);
	    y2 = b - y1;
	    GET_LDOUBLE_MSW(high,a);
	    SET_LDOUBLE_WORDS(t1,ea+1,high,0);
	    t2 = a - t1;
	    w  = sqrtl(t1*y1-(w*(-w)-(t1*y2+t2*b)));
	}
	if(k!=0) {
	    uint32_t exp;
	    t1 = 1.0;
	    GET_LDOUBLE_EXP(exp,t1);
	    SET_LDOUBLE_EXP(t1,exp+k);
	    w *= t1;
	    math_check_force_underflow_nonneg (w);
	    return w;
	} else return w;
}
libm_alias_finite (__ieee754_hypotl, __hypotl)
