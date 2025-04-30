/* s_tanhf.c -- float version of s_tanh.c.
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

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: s_tanhf.c,v 1.4 1995/05/10 20:48:24 jtc Exp $";
#endif

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <libm-alias-float.h>

static const float one=1.0, two=2.0, tiny = 1.0e-30;

float __tanhf(float x)
{
	float t,z;
	int32_t jx,ix;

	GET_FLOAT_WORD(jx,x);
	ix = jx&0x7fffffff;

    /* x is INF or NaN */
	if(ix>=0x7f800000) {
	    if (jx>=0) return one/x+one;    /* tanh(+-inf)=+-1 */
	    else       return one/x-one;    /* tanh(NaN) = NaN */
	}

    /* |x| < 22 */
	if (ix < 0x41b00000) {		/* |x|<22 */
	    if (ix == 0)
		return x;		/* x == +-0 */
	    if (ix<0x24000000) 		/* |x|<2**-55 */
	      {
		math_check_force_underflow (x);
		return x*(one+x);    	/* tanh(small) = small */
	      }
	    if (ix>=0x3f800000) {	/* |x|>=1  */
		t = __expm1f(two*fabsf(x));
		z = one - two/(t+two);
	    } else {
	        t = __expm1f(-two*fabsf(x));
	        z= -t/(t+two);
	    }
    /* |x| > 22, return +-1 */
	} else {
	    z = one - tiny;		/* raised inexact flag */
	}
	return (jx>=0)? z: -z;
}
libm_alias_float (__tanh, tanh)
