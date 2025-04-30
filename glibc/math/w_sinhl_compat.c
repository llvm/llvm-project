/* w_sinhl.c -- long double version of w_sinh.c.
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
 * wrapper sinhl(x)
 */

#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>
#include <libm-alias-ldouble.h>

#if LIBM_SVID_COMPAT
long double
__sinhl (long double x)
{
	long double z = __ieee754_sinhl (x);
	if (__builtin_expect (!isfinite (z), 0) && isfinite (x)
	    && _LIB_VERSION != _IEEE_)
	    return __kernel_standard_l (x, x, 225); /* sinh overflow */

	return z;
}
libm_alias_ldouble (__sinh, sinh)
#endif
