/* s_scalbnl.c -- long double version of s_scalbn.c.
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

/*
 * scalbnl (long double x, int n)
 * scalbnl(x,n) returns x* 2**n  computed by  exponent
 * manipulation rather than by actually performing an
 * exponentiation or a multiplication.
 */

#include <math.h>
#include <math_private.h>

static const long double
two63   =  0x1p63L,
twom64  =  0x1p-64L,
huge   = 1.0e+4900L,
tiny   = 1.0e-4900L;

long double
__scalblnl (long double x, long int n)
{
	int32_t k,es,hx,lx;
	GET_LDOUBLE_WORDS(es,hx,lx,x);
	k = es&0x7fff;				/* extract exponent */
	if (__builtin_expect(k==0, 0)) {	/* 0 or subnormal x */
	    if ((lx|(hx&0x7fffffff))==0) return x; /* +-0 */
	    x *= two63;
	    GET_LDOUBLE_EXP(es,x);
	    k = (es&0x7fff) - 63;
	    }
	if (__builtin_expect(k==0x7fff, 0)) return x+x;	/* NaN or Inf */
	if (__builtin_expect(n< -50000, 0))
	  return tiny*copysignl(tiny,x);
	if (__builtin_expect(n> 50000 || k+n > 0x7ffe, 0))
	  return huge*copysignl(huge,x); /* overflow  */
	/* Now k and n are bounded we know that k = k+n does not
	   overflow.  */
	k = k+n;
	if (__builtin_expect(k > 0, 1))		/* normal result */
	    {SET_LDOUBLE_EXP(x,(es&0x8000)|k); return x;}
	if (k <= -64)
	    return tiny*copysignl(tiny,x); 	/*underflow*/
	k += 64;				/* subnormal result */
	SET_LDOUBLE_EXP(x,(es&0x8000)|k);
	return x*twom64;
}
