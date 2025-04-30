/* @(#)w_gamma.c 5.1 93/09/24 */
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

/* double gamma(double x)
 * Return  the logarithm of the Gamma function of x or the Gamma function of x,
 * depending on the library mode.
 */

#include <errno.h>
#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>
#include <libm-alias-double.h>

#if LIBM_SVID_COMPAT
double
__tgamma(double x)
{
	int local_signgam;
	double y = __ieee754_gamma_r(x,&local_signgam);

	if(__glibc_unlikely (!isfinite (y) || y == 0)
	   && (isfinite (x) || (isinf (x) && x < 0.0))
	   && _LIB_VERSION != _IEEE_) {
	  if (x == 0.0)
	    return __kernel_standard(x,x,50); /* tgamma pole */
	  else if(floor(x)==x&&x<0.0)
	    return __kernel_standard(x,x,41); /* tgamma domain */
	  else if (y == 0)
	    __set_errno (ERANGE); /* tgamma underflow */
	  else
	    return __kernel_standard(x,x,40); /* tgamma overflow */
	}
	return local_signgam < 0 ? -y : y;
}
libm_alias_double (__tgamma, tgamma)
#endif
