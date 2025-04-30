/* s_asinhl.c -- long double version of s_asinh.c.
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

/* asinhl(x)
 * Method :
 *	Based on
 *		asinhl(x) = signl(x) * logl [ |x| + sqrtl(x*x+1) ]
 *	we have
 *	asinhl(x) := x  if  1+x*x=1,
 *		  := signl(x)*(logl(x)+ln2)) for large |x|, else
 *		  := signl(x)*logl(2|x|+1/(|x|+sqrtl(x*x+1))) if|x|>2, else
 *		  := signl(x)*log1pl(|x| + x^2/(1 + sqrtl(1+x^2)))
 */

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <libm-alias-ldouble.h>

static const long double
one =  1.000000000000000000000e+00L, /* 0x3FFF, 0x00000000, 0x00000000 */
ln2 =  6.931471805599453094287e-01L, /* 0x3FFE, 0xB17217F7, 0xD1CF79AC */
huge=  1.000000000000000000e+4900L;

long double __asinhl(long double x)
{
	long double t,w;
	int32_t hx,ix;
	GET_LDOUBLE_EXP(hx,x);
	ix = hx&0x7fff;
	if(__builtin_expect(ix< 0x3fde, 0)) {	/* |x|<2**-34 */
	    math_check_force_underflow (x);
	    if(huge+x>one) return x;	/* return x inexact except 0 */
	}
	if(__builtin_expect(ix>0x4020,0)) {		/* |x| > 2**34 */
	    if(ix==0x7fff) return x+x;	/* x is inf or NaN */
	    w = __ieee754_logl(fabsl(x))+ln2;
	} else {
	    long double xa = fabsl(x);
	    if (ix>0x4000) {	/* 2**34 > |x| > 2.0 */
		w = __ieee754_logl(2.0*xa+one/(sqrtl(xa*xa+one)+xa));
	    } else {		/* 2.0 > |x| > 2**-28 */
		t = xa*xa;
		w =__log1pl(xa+t/(one+sqrtl(one+t)));
	    }
	}
	return copysignl(w, x);
}
libm_alias_ldouble (__asinh, asinh)
