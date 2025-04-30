/* w_gammal.c -- long double version of w_gamma.c.
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

/* long double gammal(double x)
 * Return the Gamma function of x.
 */

#include <errno.h>
#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>
#include <libm-alias-ldouble.h>

#if LIBM_SVID_COMPAT
long double
__tgammal(long double x)
{
	int local_signgam;
	long double y = __ieee754_gammal_r(x,&local_signgam);

	if(__glibc_unlikely (!isfinite (y) || y == 0)
	   && (isfinite (x) || (isinf (x) && x < 0.0))
	   && _LIB_VERSION != _IEEE_) {
	  if(x==0.0)
	    return __kernel_standard_l(x,x,250); /* tgamma pole */
	  else if(floorl(x)==x&&x<0.0L)
	    return __kernel_standard_l(x,x,241); /* tgamma domain */
	  else if (y == 0)
	    __set_errno (ERANGE); /* tgamma underflow */
	  else
	    return __kernel_standard_l(x,x,240); /* tgamma overflow */
	}
	return local_signgam < 0 ? - y : y;
}
libm_alias_ldouble (__tgamma, tgamma)
#endif
