/* s_modfl.c -- long double version of s_modf.c.
 * Conversion to IEEE quad long double by Jakub Jelinek, jj@ultra.linux.cz.
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
 * modfl(long double x, long double *iptr)
 * return fraction part of x, and return x's integral part in *iptr.
 * Method:
 *	Bit twiddling.
 *
 * Exception:
 *	No exception.
 */

#include <math.h>
#include <math_private.h>
#include <math_ldbl_opt.h>

long double __modfl(long double x, long double *iptr)
{
	int64_t i0,i1,j0;
	uint64_t i;
	double xhi, xlo;

	ldbl_unpack (x, &xhi, &xlo);
	EXTRACT_WORDS64 (i0, xhi);
	EXTRACT_WORDS64 (i1, xlo);
	i1 &= 0x000fffffffffffffLL;
	j0 = ((i0>>52)&0x7ff)-0x3ff;	/* exponent of x */
	if(j0<52) {			/* integer part in high x */
	    if(j0<0) {			/* |x|<1 */
		/* *iptr = +-0 */
		INSERT_WORDS64 (xhi, i0&0x8000000000000000ULL);
		*iptr = xhi;
		return x;
	    } else {
		i = (0x000fffffffffffffLL)>>j0;
		if(((i0&i)|(i1&0x7fffffffffffffffLL))==0) {		/* x is integral */
		    *iptr = x;
		    /* return +-0 */
		    INSERT_WORDS64 (xhi, i0&0x8000000000000000ULL);
		    x = xhi;
		    return x;
		} else {
		    INSERT_WORDS64 (xhi, i0&(~i));
		    *iptr = xhi;
		    return x - *iptr;
		}
	    }
	} else if (j0>103) {		/* no fraction part */
	    *iptr = x;
	    /* We must handle NaNs separately.  */
	    if ((i0 & 0x7fffffffffffffffLL) > 0x7ff0000000000000LL)
	      return *iptr = x + x;
	    /* return +-0 */
	    INSERT_WORDS64 (xhi, i0&0x8000000000000000ULL);
	    x = xhi;
	    return x;
	} else {			/* fraction part in low x */
	    i = -1ULL>>(j0-52);
	    if((i1&i)==0) { 		/* x is integral */
		*iptr = x;
		/* return +-0 */
		INSERT_WORDS64 (xhi, i0&0x8000000000000000ULL);
		x = xhi;
		return x;
	    } else {
		INSERT_WORDS64 (xhi, i0);
		INSERT_WORDS64 (xlo, i1&(~i));
		*iptr = ldbl_pack (xhi, xlo);
		return x - *iptr;
	    }
	}
}
#if IS_IN (libm)
long_double_symbol (libm, __modfl, modfl);
#else
long_double_symbol (libc, __modfl, modfl);
#endif
