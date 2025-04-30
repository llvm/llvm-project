/* s_atanhl.c -- long double version of s_atan.c.
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

/* __ieee754_atanhl(x)
 * Method :
 *    1.Reduced x to positive by atanh(-x) = -atanh(x)
 *    2.For x>=0.5
 *                   1              2x                          x
 *	atanhl(x) = --- * log(1 + -------) = 0.5 * log1p(2 * --------)
 *                   2             1 - x                      1 - x
 *
 *	For x<0.5
 *	atanhl(x) = 0.5*log1pl(2x+2x*x/(1-x))
 *
 * Special cases:
 *	atanhl(x) is NaN if |x| > 1 with signal;
 *	atanhl(NaN) is that NaN with no signal;
 *	atanhl(+-1) is +-INF with signal.
 *
 */

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <libm-alias-finite.h>

static const _Float128 one = 1, huge = L(1e4900);

static const _Float128 zero = 0;

_Float128
__ieee754_atanhl(_Float128 x)
{
	_Float128 t;
	uint32_t jx, ix;
	ieee854_long_double_shape_type u;

	u.value = x;
	jx = u.parts32.w0;
	ix = jx & 0x7fffffff;
	u.parts32.w0 = ix;
	if (ix >= 0x3fff0000) /* |x| >= 1.0 or infinity or NaN */
	  {
	    if (u.value == one)
	      return x/zero;
	    else
	      return (x-x)/(x-x);
	  }
	if(ix<0x3fc60000 && (huge+x)>zero)	/* x < 2^-57 */
	  {
	    math_check_force_underflow (x);
	    return x;
	  }

	if(ix<0x3ffe0000) {		/* x < 0.5 */
	    t = u.value+u.value;
	    t = 0.5*__log1pl(t+t*u.value/(one-u.value));
	} else
	    t = 0.5*__log1pl((u.value+u.value)/(one-u.value));
	if(jx & 0x80000000) return -t; else return t;
}
libm_alias_finite (__ieee754_atanhl, __atanhl)
