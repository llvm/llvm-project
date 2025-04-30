/* @(#)wr_lgamma.c 5.1 93/09/24 */
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
 * wrapper double lgamma_r(double x, int *signgamp)
 */

#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>
#include <libm-alias-double.h>


#if LIBM_SVID_COMPAT
double
__lgamma_r(double x, int *signgamp)
{
	double y = __ieee754_lgamma_r(x,signgamp);
	if(__builtin_expect(!isfinite(y), 0)
	   && isfinite(x) && _LIB_VERSION != _IEEE_)
		return __kernel_standard(x, x,
					 floor(x)==x&&x<=0.0
					 ? 15 /* lgamma pole */
					 : 14); /* lgamma overflow */

	return y;
}
libm_alias_double_r (__lgamma, lgamma, _r)
#endif
