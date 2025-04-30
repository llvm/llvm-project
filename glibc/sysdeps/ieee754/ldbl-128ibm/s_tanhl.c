/* @(#)s_tanh.c 5.1 93/09/24 */
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
static char rcsid[] = "$NetBSD: s_tanh.c,v 1.7 1995/05/10 20:48:22 jtc Exp $";
#endif

/* Tanh(x)
 * Return the Hyperbolic Tangent of x
 *
 * Method :
 *				       x    -x
 *				      e  - e
 *	0. tanh(x) is defined to be -----------
 *				       x    -x
 *				      e  + e
 *	1. reduce x to non-negative by tanh(-x) = -tanh(x).
 *	2.  0      <= x <= 2**-57 : tanh(x) := x*(one+x)
 *					        -t
 *	    2**-57 <  x <=  1     : tanh(x) := -----; t = expm1(-2x)
 *					       t + 2
 *						     2
 *	    1      <= x <=  40.0  : tanh(x) := 1-  ----- ; t=expm1(2x)
 *						   t + 2
 *	    40.0   <  x <= INF    : tanh(x) := 1.
 *
 * Special cases:
 *	tanh(NaN) is NaN;
 *	only tanh(0)=0 is exact for finite argument.
 */

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <math_ldbl_opt.h>

static const long double one=1.0L, two=2.0L, tiny = 1.0e-300L;

long double __tanhl(long double x)
{
	long double t,z;
	int64_t jx,ix;
	double xhi;

    /* High word of |x|. */
	xhi = ldbl_high (x);
	EXTRACT_WORDS64 (jx, xhi);
	ix = jx&0x7fffffffffffffffLL;

    /* x is INF or NaN */
	if(ix>=0x7ff0000000000000LL) {
	    if (jx>=0) return one/x+one;    /* tanh(+-inf)=+-1 */
	    else       return one/x-one;    /* tanh(NaN) = NaN */
	}

    /* |x| < 40 */
	if (ix < 0x4044000000000000LL) {		/* |x|<40 */
	    if (ix == 0)
		return x;		/* x == +-0 */
	    if (ix<0x3c60000000000000LL) 	/* |x|<2**-57 */
	      {
		math_check_force_underflow (x);
		return x;		/* tanh(small) = small */
	      }
	    if (ix>=0x3ff0000000000000LL) {	/* |x|>=1  */
		t = __expm1l(two*fabsl(x));
		z = one - two/(t+two);
	    } else {
	        t = __expm1l(-two*fabsl(x));
	        z= -t/(t+two);
	    }
    /* |x| > 40, return +-1 */
	} else {
	    z = one - tiny;		/* raised inexact flag */
	}
	return (jx>=0)? z: -z;
}
long_double_symbol (libm, __tanhl, tanhl);
