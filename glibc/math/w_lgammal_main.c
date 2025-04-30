/* w_lgammal.c -- long double version of w_lgamma.c.
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

/* long double lgammal(long double x)
 * Return the logarithm of the Gamma function of x.
 *
 * Method: call __ieee754_lgammal_r
 */

#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>
#include <libm-alias-ldouble.h>

#include <lgamma-compat.h>

#if BUILD_LGAMMA
long double
LGFUNC (__lgammal) (long double x)
{
	long double y = CALL_LGAMMA (long double, __ieee754_lgammal_r, x);
	if(__builtin_expect(!isfinite(y), 0)
	   && isfinite(x) && _LIB_VERSION != _IEEE_)
		return __kernel_standard_l(x, x,
					   floorl(x)==x&&x<=0.0L
					   ? 215 /* lgamma pole */
					   : 214); /* lgamma overflow */

	return y;
}
# if USE_AS_COMPAT
compat_symbol (libm, __lgammal_compat, lgammal, LGAMMA_OLD_VER);
# else
versioned_symbol (libm, __lgammal, lgammal, LGAMMA_NEW_VER);
libm_alias_ldouble_other (__lgamma, lgamma)
# endif
# if GAMMA_ALIAS
strong_alias (LGFUNC (__lgammal), __gammal)
weak_alias (__gammal, gammal)
# endif
#endif
