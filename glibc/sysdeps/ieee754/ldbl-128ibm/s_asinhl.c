/* @(#)s_asinh.c 5.1 93/09/24 */
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
static char rcsid[] = "$NetBSD: s_asinh.c,v 1.9 1995/05/12 04:57:37 jtc Exp $";
#endif

/* asinh(x)
 * Method :
 *	Based on
 *		asinh(x) = sign(x) * log [ |x| + sqrt(x*x+1) ]
 *	we have
 *	asinh(x) := x  if  1+x*x=1,
 *		 := sign(x)*(log(x)+ln2)) for large |x|, else
 *		 := sign(x)*log(2|x|+1/(|x|+sqrt(x*x+1))) if|x|>2, else
 *		 := sign(x)*log1p(|x| + x^2/(1 + sqrt(1+x^2)))
 */

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <math_ldbl_opt.h>

static const long double
one =  1.00000000000000000000e+00L, /* 0x3ff0000000000000, 0 */
ln2 =  0.6931471805599453094172321214581766L, /* 0x3fe62e42fefa39ef, 0x3c7abc9e3b398040 */
huge=  1.00000000000000000000e+300L;

long double __asinhl(long double x)
{
	long double t,w;
	int64_t hx,ix;
	double xhi;

	xhi = ldbl_high (x);
	EXTRACT_WORDS64 (hx, xhi);
	ix = hx&0x7fffffffffffffffLL;
	if(ix>=0x7ff0000000000000LL) return x+x;	/* x is inf or NaN */
	if(ix< 0x3c70000000000000LL) {	/* |x|<2**-56 */
	    math_check_force_underflow (x);
	    if(huge+x>one) return x;	/* return x inexact except 0 */
	}
	if(ix>0x4370000000000000LL) {	/* |x| > 2**56 */
	    w = __ieee754_logl(fabsl(x))+ln2;
	} else if (ix>0x4000000000000000LL) {	/* 2**56 >= |x| > 2.0 */
	    t = fabs(x);
	    w = __ieee754_logl(2.0*t+one/(sqrtl(x*x+one)+t));
	} else {		/* 2.0 >= |x| >= 2**-56 */
	    t = x*x;
	    w =__log1pl(fabsl(x)+t/(one+sqrtl(one+t)));
	}
	if(hx>0) return w; else return -w;
}
long_double_symbol (libm, __asinhl, asinhl);
