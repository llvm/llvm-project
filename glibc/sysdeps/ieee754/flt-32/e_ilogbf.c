/* s_ilogbf.c -- float version of s_ilogb.c.
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
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
static char rcsid[] = "$NetBSD: s_ilogbf.c,v 1.4 1995/05/10 20:47:31 jtc Exp $";
#endif

#include <limits.h>
#include <math.h>
#include <math_private.h>

int __ieee754_ilogbf(float x)
{
	int32_t hx,ix;

	GET_FLOAT_WORD(hx,x);
	hx &= 0x7fffffff;
	if(hx<0x00800000) {
	    if(hx==0)
		return FP_ILOGB0;	/* ilogb(0) = FP_ILOGB0 */
	    else			/* subnormal x */
	        for (ix = -126,hx<<=8; hx>0; hx<<=1) ix -=1;
	    return ix;
	}
	else if (hx<0x7f800000) return (hx>>23)-127;
	else if (FP_ILOGBNAN != INT_MAX) {
	    /* ISO C99 requires ilogbf(+-Inf) == INT_MAX.  */
	    if (hx==0x7f800000)
		return INT_MAX;
	}
	return FP_ILOGBNAN;
}
