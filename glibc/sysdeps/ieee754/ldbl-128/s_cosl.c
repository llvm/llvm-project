/* s_cosl.c -- long double version of s_cos.c.
 * Conversion to long double by Jakub Jelinek, jj@ultra.linux.cz.
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

/* cosl(x)
 * Return cosine function of x.
 *
 * kernel function:
 *	__kernel_sinl		... sine function on [-pi/4,pi/4]
 *	__kernel_cosl		... cosine function on [-pi/4,pi/4]
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

_Float128 __cosl(_Float128 x)
{
	_Float128 y[2],z=0;
	int64_t n, ix;

    /* High word of x. */
	GET_LDOUBLE_MSW64(ix,x);

    /* |x| ~< pi/4 */
	ix &= 0x7fffffffffffffffLL;
	if(ix <= 0x3ffe921fb54442d1LL)
	  return __kernel_cosl(x,z);

    /* cos(Inf or NaN) is NaN */
	else if (ix>=0x7fff000000000000LL) {
	    if (ix == 0x7fff000000000000LL) {
		GET_LDOUBLE_LSW64(n,x);
		if (n == 0)
		    __set_errno (EDOM);
	    }
	    return x-x;
	}

    /* argument reduction needed */
	else {
	    n = __ieee754_rem_pio2l(x,y);
	    switch(n&3) {
		case 0: return  __kernel_cosl(y[0],y[1]);
		case 1: return -__kernel_sinl(y[0],y[1],1);
		case 2: return -__kernel_cosl(y[0],y[1]);
		default:
		        return  __kernel_sinl(y[0],y[1],1);
	    }
	}
}
libm_alias_ldouble (__cos, cos)
