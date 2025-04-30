/* s_scalbnf.c -- float version of s_scalbn.c.
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
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

#include <math.h>
#include <math_private.h>

static const float
two25   =  3.355443200e+07,	/* 0x4c000000 */
twom25  =  2.9802322388e-08,	/* 0x33000000 */
huge   = 1.0e+30,
tiny   = 1.0e-30;

float
__scalbnf (float x, int n)
{
	int32_t k,ix;
	GET_FLOAT_WORD(ix,x);
	k = (ix&0x7f800000)>>23;		/* extract exponent */
	if (__builtin_expect(k==0, 0)) {	/* 0 or subnormal x */
	    if ((ix&0x7fffffff)==0) return x; /* +-0 */
	    x *= two25;
	    GET_FLOAT_WORD(ix,x);
	    k = ((ix&0x7f800000)>>23) - 25;
	    }
	if (__builtin_expect(k==0xff, 0)) return x+x;	/* NaN or Inf */
	if (__builtin_expect(n< -50000, 0))
	  return tiny*copysignf(tiny,x);	/*underflow*/
	if (__builtin_expect(n> 50000 || k+n > 0xfe, 0))
	  return huge*copysignf(huge,x); /* overflow  */
	/* Now k and n are bounded we know that k = k+n does not
	   overflow.  */
	k = k+n;
	if (__builtin_expect(k > 0, 1))		/* normal result */
	    {SET_FLOAT_WORD(x,(ix&0x807fffff)|(k<<23)); return x;}
	if (k <= -25)
	    return tiny*copysignf(tiny,x);	/*underflow*/
	k += 25;				/* subnormal result */
	SET_FLOAT_WORD(x,(ix&0x807fffff)|(k<<23));
	return x*twom25;
}
