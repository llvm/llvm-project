/* s_frexpl.c -- long double version of s_frexp.c.
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

/*
 * for non-zero x
 *	x = frexpl(arg,&exp);
 * return a long double fp quantity x such that 0.5 <= |x| <1.0
 * and the corresponding binary exponent "exp". That is
 *	arg = x*2^exp.
 * If arg is inf, 0.0, or NaN, then frexpl(arg,&exp) returns arg
 * with *exp=0.
 */

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <libm-alias-ldouble.h>

static const long double
#if LDBL_MANT_DIG == 64
two65 =  3.68934881474191032320e+19L; /* 0x4040, 0x80000000, 0x00000000 */
#else
# error "Cannot handle this MANT_DIG"
#endif


long double __frexpl(long double x, int *eptr)
{
	uint32_t se, hx, ix, lx;
	GET_LDOUBLE_WORDS(se,hx,lx,x);
	ix = 0x7fff&se;
	*eptr = 0;
	if(ix==0x7fff||((ix|hx|lx)==0)) return x + x;	/* 0,inf,nan */
	if (ix==0x0000) {		/* subnormal */
	    x *= two65;
	    GET_LDOUBLE_EXP(se,x);
	    ix = se&0x7fff;
	    *eptr = -65;
	}
	*eptr += ix-16382;
	se = (se & 0x8000) | 0x3ffe;
	SET_LDOUBLE_EXP(x,se);
	return x;
}
libm_alias_ldouble (__frexp, frexp)
