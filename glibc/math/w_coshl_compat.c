/* w_acoshl.c -- long double version of w_acosh.c.
 * Conversion to long double by Ulrich Drepper,
 * Cygnus Support, drepper@cygnus.com.
 * Optimizations by Ulrich Drepper <drepper@gmail.com>, 2011.
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
 * wrapper coshl(x)
 */

#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>
#include <libm-alias-ldouble.h>

#if LIBM_SVID_COMPAT
long double
__coshl (long double x)
{
	long double z = __ieee754_coshl (x);
	if (__builtin_expect (!isfinite (z), 0) && isfinite (x)
	    && _LIB_VERSION != _IEEE_)
		return __kernel_standard_l (x, x, 205); /* cosh overflow */

	return z;
}
libm_alias_ldouble (__cosh, cosh)
#endif
