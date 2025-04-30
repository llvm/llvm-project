/* s_isnanl.c -- long double version for i387 of s_isnan.c.
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
 * isnanl(x) returns 1 is x is nan, else 0;
 * no branching!
 */

#include <math.h>
#include <math_private.h>

int __isnanl(long double x)
{
	int32_t se,hx,lx,pn;
	GET_LDOUBLE_WORDS(se,hx,lx,x);
	se = (se & 0x7fff) << 1;
	/* Detect pseudo-normal numbers, i.e. exponent is non-zero and the top
	   bit of the significand is not set.   */
	pn = (uint32_t)((~hx & 0x80000000) & (se|(-se)))>>31;
	/* Clear the significand bit when computing mantissa.  */
	lx |= hx & 0x7fffffff;
	se |= (uint32_t)(lx|(-lx))>>31;
	se = 0xfffe - se;

	return (int)(((uint32_t)(se)) >> 16) | pn;
}
hidden_def (__isnanl)
weak_alias (__isnanl, isnanl)
