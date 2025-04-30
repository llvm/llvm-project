/* s_tanl.c -- long double version of s_tan.c.
 * Conversion to IEEE quad long double by Jakub Jelinek, jj@ultra.linux.cz.
 */

/* @(#)s_tan.c 5.1 93/09/24 */
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

/* tanl(x)
 * Return tangent function of x.
 *
 * kernel function:
 *	__kernel_tanl		... tangent function on [-pi/4,pi/4]
 *	__ieee754_rem_pio2l	... argument reduction routine
 *
 * Method.
 *      Let S,C and T denote the sin, cos and tan respectively on
 *	[-PI/4, +PI/4]. Reduce the argument x to y1+y2 = x-k*pi/2
 *	in [-pi/4 , +pi/4], and let n = k mod 4.
 *	We have
 *
 *          n        sin(x)      cos(x)        tan(x)
 *     ----------------------------------------------------------
 *	    0	       S	   C		 T
 *	    1	       C	  -S		-1/T
 *	    2	      -S	  -C		 T
 *	    3	      -C	   S		-1/T
 *     ----------------------------------------------------------
 *
 * Special cases:
 *      Let trig be any of sin, cos, or tan.
 *      trig(+-INF)  is NaN, with signals;
 *      trig(NaN)    is that NaN;
 *
 * Accuracy:
 *	TRIG(x) returns trig(x) nearly rounded
 */

#include <errno.h>
#include <math.h>
#include <math_private.h>
#include <math_ldbl_opt.h>

long double __tanl(long double x)
{
	long double y[2],z=0.0L;
	int64_t n, ix;
	double xhi;

    /* High word of x. */
	xhi = ldbl_high (x);
	EXTRACT_WORDS64 (ix, xhi);

    /* |x| ~< pi/4 */
	ix &= 0x7fffffffffffffffLL;
	if(ix <= 0x3fe921fb54442d10LL) return __kernel_tanl(x,z,1);

    /* tanl(Inf or NaN) is NaN */
	else if (ix>=0x7ff0000000000000LL) {
	    if (ix == 0x7ff0000000000000LL)
		__set_errno (EDOM);
	    return x-x;		/* NaN */
	}
    /* argument reduction needed */
	else {
	    n = __ieee754_rem_pio2l(x,y);
	    return __kernel_tanl(y[0],y[1],1-((n&1)<<1)); /*   1 -- n even
							-1 -- n odd */
	}
}
long_double_symbol (libm, __tanl, tanl);
