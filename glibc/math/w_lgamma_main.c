/* @(#)w_lgamma.c 5.1 93/09/24 */
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

/* double lgamma(double x)
 * Return the logarithm of the Gamma function of x.
 *
 * Method: call __ieee754_lgamma_r
 */

#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>
#include <libm-alias-double.h>

#include <lgamma-compat.h>

#if BUILD_LGAMMA
double
LGFUNC (__lgamma) (double x)
{
	double y = CALL_LGAMMA (double, __ieee754_lgamma_r, x);
	if(__builtin_expect(!isfinite(y), 0)
	   && isfinite(x) && _LIB_VERSION != _IEEE_)
		return __kernel_standard(x, x,
					 floor(x)==x&&x<=0.0
					 ? 15 /* lgamma pole */
					 : 14); /* lgamma overflow */

	return y;
}
# if USE_AS_COMPAT
compat_symbol (libm, __lgamma_compat, lgamma, LGAMMA_OLD_VER);
#  ifdef NO_LONG_DOUBLE
strong_alias (__lgamma_compat, __lgammal_compat)
compat_symbol (libm, __lgammal_compat, lgammal, LGAMMA_OLD_VER);
#  endif
# else
versioned_symbol (libm, __lgamma, lgamma, LGAMMA_NEW_VER);
#  ifdef NO_LONG_DOUBLE
strong_alias (__lgamma, __lgammal)
versioned_symbol (libm, __lgammal, lgammal, LGAMMA_NEW_VER);
#  endif
libm_alias_double_other (__lgamma, lgamma)
# endif
# if GAMMA_ALIAS
strong_alias (LGFUNC (__lgamma), __gamma)
weak_alias (__gamma, gamma)
#  ifdef NO_LONG_DOUBLE
strong_alias (__gamma, __gammal)
weak_alias (__gamma, gammal)
#  endif
# endif
#endif
