/* e_log10f.c -- float version of e_log10.c.
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
#include <fix-int-fp-convert-zero.h>
#include <libm-alias-finite.h>

static const float
two25      =  3.3554432000e+07, /* 0x4c000000 */
ivln10     =  4.3429449201e-01, /* 0x3ede5bd9 */
log10_2hi  =  3.0102920532e-01, /* 0x3e9a2080 */
log10_2lo  =  7.9034151668e-07; /* 0x355427db */

float
__ieee754_log10f(float x)
{
	float y,z;
	int32_t i,k,hx;

	GET_FLOAT_WORD(hx,x);

	k=0;
	if (hx < 0x00800000) {			/* x < 2**-126  */
	    if (__builtin_expect((hx&0x7fffffff)==0, 0))
	      return -two25/fabsf (x);	/* log(+-0)=-inf  */
	    if (__builtin_expect(hx<0, 0))
		return (x-x)/(x-x);	/* log(-#) = NaN */
	    k -= 25; x *= two25; /* subnormal number, scale up x */
	    GET_FLOAT_WORD(hx,x);
	}
	if (__builtin_expect(hx >= 0x7f800000, 0)) return x+x;
	k += (hx>>23)-127;
	i  = ((uint32_t)k&0x80000000)>>31;
	hx = (hx&0x007fffff)|((0x7f-i)<<23);
	y  = (float)(k+i);
	if (FIX_INT_FP_CONVERT_ZERO && y == 0.0f)
	  y = 0.0f;
	SET_FLOAT_WORD(x,hx);
	z  = y*log10_2lo + ivln10*__ieee754_logf(x);
	return  z+y*log10_2hi;
}
libm_alias_finite (__ieee754_log10f, __log10f)
