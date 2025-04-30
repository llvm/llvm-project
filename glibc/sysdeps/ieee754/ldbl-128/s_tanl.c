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
#include <libm-alias-ldouble.h>

_Float128 __tanl(_Float128 x)
{
	_Float128 y[2],z=0;
	int64_t n, ix;

    /* High word of x. */
	GET_LDOUBLE_MSW64(ix,x);

    /* |x| ~< pi/4 */
	ix &= 0x7fffffffffffffffLL;
	if(ix <= 0x3ffe921fb54442d1LL) return __kernel_tanl(x,z,1);

    /* tanl(Inf or NaN) is NaN */
	else if (ix>=0x7fff000000000000LL) {
	    if (ix == 0x7fff000000000000LL) {
		GET_LDOUBLE_LSW64(n,x);
		if (n == 0)
		    __set_errno (EDOM);
	    }
	    return x-x;		/* NaN */
	}

    /* argument reduction needed */
	else {
	    n = __ieee754_rem_pio2l(x,y);
	    return __kernel_tanl(y[0],y[1],1-((n&1)<<1)); /*   1 -- n even
							-1 -- n odd */
	}
}
libm_alias_ldouble (__tan, tan)
