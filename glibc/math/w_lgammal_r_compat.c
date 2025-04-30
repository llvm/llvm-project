/* w_lgammal_r.c -- long double version of w_lgamma_r.c.
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
 * wrapper long double lgammal_r(long double x, int *signgamp)
 */

#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>
#include <libm-alias-ldouble.h>


#if LIBM_SVID_COMPAT
long double
__lgammal_r(long double x, int *signgamp)
{
	long double y = __ieee754_lgammal_r(x,signgamp);
	if(__builtin_expect(!isfinite(y), 0)
	   && isfinite(x) && _LIB_VERSION != _IEEE_)
		return __kernel_standard(x, x,
					 floorl(x)==x&&x<=0.0
					 ? 215 /* lgamma pole */
					 : 214); /* lgamma overflow */

	return y;
}
libm_alias_ldouble_r (__lgamma, lgamma, _r)
#endif
