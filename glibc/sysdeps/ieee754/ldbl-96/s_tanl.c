/* s_tanl.c -- long double version of s_tan.c.
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

long double __tanl(long double x)
{
	long double y[2],z=0.0;
	int32_t n, se, i0, i1;

    /* High word of x. */
	GET_LDOUBLE_WORDS(se,i0,i1,x);

    /* |x| ~< pi/4 */
	se &= 0x7fff;
	if(se <= 0x3ffe) return __kernel_tanl(x,z,1);

    /* tan(Inf or NaN) is NaN */
	else if (se==0x7fff) {
	  if (i1 == 0 && i0 == 0x80000000)
	    __set_errno (EDOM);
	  return x-x;
	}

    /* argument reduction needed */
	else {
	    n = __ieee754_rem_pio2l(x,y);
	    return __kernel_tanl(y[0],y[1],1-((n&1)<<1)); /*   1 -- n even
							-1 -- n odd */
	}
}
libm_alias_ldouble (__tan, tan)
