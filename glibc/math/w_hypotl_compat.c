/* w_hypotl.c -- long double version of w_hypot.c.
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
 * wrapper hypotl(x,y)
 */

#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>
#include <libm-alias-ldouble.h>


#if LIBM_SVID_COMPAT
long double
__hypotl(long double x, long double y)
{
	long double z;
	z = __ieee754_hypotl(x,y);
	if(__builtin_expect(!isfinite(z), 0)
	   && isfinite(x) && isfinite(y) && _LIB_VERSION != _IEEE_)
	    return __kernel_standard_l(x, y, 204); /* hypot overflow */

	return z;
}
libm_alias_ldouble (__hypot, hypot)
#endif
