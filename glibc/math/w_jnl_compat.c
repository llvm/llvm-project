/* w_jnl.c -- long double version of w_jn.c.
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
 * wrapper jn(int n, double x), yn(int n, double x)
 * floating point Bessel's function of the 1st and 2nd kind
 * of order n
 *
 * Special cases:
 *	y0(0)=y1(0)=yn(n,0) = -inf with division by zero signal;
 *	y0(-ve)=y1(-ve)=yn(n,-ve) are NaN with invalid signal.
 * Note 2. About jn(n,x), yn(n,x)
 *	For n=0, j0(x) is called,
 *	for n=1, j1(x) is called,
 *	for n<x, forward recursion us used starting
 *	from values of j0(x) and j1(x).
 *	for n>x, a continued fraction approximation to
 *	j(n,x)/j(n-1,x) is evaluated and then backward
 *	recursion is used starting from a supposed value
 *	for j(n,x). The resulting value of j(0,x) is
 *	compared with the actual value to correct the
 *	supposed value of j(n,x).
 *
 *	yn(n,x) is similar in all respects, except
 *	that forward recursion is used for all
 *	values of n>1.
 *
 */

#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>
#include <libm-alias-ldouble.h>

#if LIBM_SVID_COMPAT
long double __jnl(int n, long double x)	/* wrapper jnl */
{
# ifdef _IEEE_LIBM
	return __ieee754_jnl(n,x);
# else
	long double z;
	z = __ieee754_jnl(n,x);
	if (_LIB_VERSION == _IEEE_
	    || _LIB_VERSION == _POSIX_
	    || isnan(x))
	  return z;
	if(fabsl(x)>X_TLOSS) {
	    return __kernel_standard_l((double)n,x,238); /* jn(|x|>X_TLOSS,n) */
	} else
	    return z;
# endif
}
libm_alias_ldouble (__jn, jn)

long double __ynl(int n, long double x)	/* wrapper ynl */
{
# ifdef _IEEE_LIBM
	return __ieee754_ynl(n,x);
# else
	long double z;
	z = __ieee754_ynl(n,x);
	if(_LIB_VERSION == _IEEE_ || isnan(x) ) return z;
        if(x <= 0.0){
                if(x==0.0)
                    /* d= -one/(x-x); */
                    return __kernel_standard_l((double)n,x,212);
                else
                    /* d = zero/(x-x); */
                    return __kernel_standard_l((double)n,x,213);
        }
	if(x>X_TLOSS && _LIB_VERSION != _POSIX_) {
	    return __kernel_standard_l((double)n,x,239); /* yn(x>X_TLOSS,n) */
	} else
	    return z;
# endif
}
libm_alias_ldouble (__yn, yn)
#endif
