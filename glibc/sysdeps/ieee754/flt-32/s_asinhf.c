/* s_asinhf.c -- float version of s_asinh.c.
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

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <libm-alias-float.h>

static const float
one =  1.0000000000e+00, /* 0x3F800000 */
ln2 =  6.9314718246e-01, /* 0x3f317218 */
huge=  1.0000000000e+30;

float
__asinhf(float x)
{
	float w;
	int32_t hx,ix;
	GET_FLOAT_WORD(hx,x);
	ix = hx&0x7fffffff;
	if(__builtin_expect(ix< 0x38000000, 0)) {	/* |x|<2**-14 */
	    math_check_force_underflow (x);
	    if(huge+x>one) return x;	/* return x inexact except 0 */
	}
	if(__builtin_expect(ix>0x47000000, 0)) {	/* |x| > 2**14 */
	    if(ix>=0x7f800000) return x+x;	/* x is inf or NaN */
	    w = __ieee754_logf(fabsf(x))+ln2;
	} else {
	    float xa = fabsf(x);
	    if (ix>0x40000000) {	/* 2**14 > |x| > 2.0 */
		w = __ieee754_logf(2.0f*xa+one/(sqrtf(xa*xa+one)+xa));
	    } else {		/* 2.0 > |x| > 2**-14 */
		float t = xa*xa;
		w =__log1pf(xa+t/(one+sqrtf(one+t)));
	    }
	}
	return copysignf(w, x);
}
libm_alias_float (__asinh, asinh)
