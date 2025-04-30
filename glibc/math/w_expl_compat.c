/* w_expl.c -- long double version of w_exp.c.
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
 * wrapper expl(x)
 */

#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>
#include <libm-alias-ldouble.h>

#if LIBM_SVID_COMPAT
long double __expl(long double x)	/* wrapper exp */
{
# ifdef _IEEE_LIBM
	return __ieee754_expl(x);
# else
	long double z = __ieee754_expl (x);
	if (__glibc_unlikely (!isfinite (z) || z == 0)
	    && isfinite (x) && _LIB_VERSION != _IEEE_)
	  return __kernel_standard_l (x, x, 206 + !!signbit (x));

	return z;
# endif
}
libm_hidden_def (__expl)
libm_alias_ldouble (__exp, exp)
#endif
