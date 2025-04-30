/* s_ilogbl.c -- long double version of s_ilogb.c.
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

/* ilogbl(long double x)
 * return the binary exponent of non-zero x
 * ilogbl(0) = FP_ILOGB0
 * ilogbl(NaN) = FP_ILOGBNAN (no signal is raised)
 * ilogbl(+-Inf) = INT_MAX (no signal is raised)
 */

#include <limits.h>
#include <math.h>
#include <math_private.h>
#include <math_ldbl_opt.h>

int __ieee754_ilogbl(long double x)
{
	int64_t hx, hxs;
	int ix;
	double xhi, xlo;

	ldbl_unpack (x, &xhi, &xlo);
	EXTRACT_WORDS64 (hx, xhi);
	hxs = hx;
	hx &= 0x7fffffffffffffffLL;
	if(hx <= 0x0010000000000000LL) {
	    if(hx==0)
		return FP_ILOGB0;	/* ilogbl(0) = FP_ILOGB0 */
	    else			/* subnormal x */
		for (ix = -1022, hx<<=11; hx>0; hx<<=1) ix -=1;
	    return ix;
	}
	else if (hx < 0x7ff0000000000000LL)
	  {
	    int hexp = (hx >> 52) - 0x3ff;
	    /* If the high part is a power of 2, and the low part is
	       nonzero with the opposite sign, the low part affects
	       the exponent.  */
	    if ((hx & 0x000fffffffffffffLL) == 0)
	      {
		int64_t lx;
		EXTRACT_WORDS64 (lx, xlo);
		if ((hxs ^ lx) < 0 && (lx & 0x7fffffffffffffffLL) != 0)
		  hexp--;
	      }
	    return hexp;
	  }
	else if (FP_ILOGBNAN != INT_MAX) {
	    /* ISO C99 requires ilogbl(+-Inf) == INT_MAX.  */
	    if (hx==0x7ff0000000000000LL)
		return INT_MAX;
	}
	return FP_ILOGBNAN;
}
