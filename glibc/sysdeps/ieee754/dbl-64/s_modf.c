/* Rewritten for 64-bit machines by Ulrich Drepper <drepper@gmail.com>.  */
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
 * modf(double x, double *iptr)
 * return fraction part of x, and return x's integral part in *iptr.
 * Method:
 *	Bit twiddling.
 *
 * Exception:
 *	No exception.
 */

#include <math.h>
#include <math_private.h>
#include <libm-alias-double.h>
#include <stdint.h>

double
__modf(double x, double *iptr)
{
	int64_t i0;
	int32_t j0;
	EXTRACT_WORDS64(i0,x);
	j0 = ((i0>>52)&0x7ff)-0x3ff;	/* exponent of x */
	if(j0<52) {			/* integer part in x */
	    if(j0<0) {			/* |x|<1 */
		/* *iptr = +-0 */
		INSERT_WORDS64(*iptr,i0&UINT64_C(0x8000000000000000));
		return x;
	    } else {
		uint64_t i = UINT64_C(0x000fffffffffffff)>>j0;
		if((i0&i)==0) {		/* x is integral */
		    *iptr = x;
		    /* return +-0 */
		    INSERT_WORDS64(x,i0&UINT64_C(0x8000000000000000));
		    return x;
		} else {
		    INSERT_WORDS64(*iptr,i0&(~i));
		    return x - *iptr;
		}
	    }
	} else { /* no fraction part */
	    *iptr = x;
	    /* We must handle NaNs separately.  */
	    if (j0 == 0x400 && (i0 & UINT64_C(0xfffffffffffff)))
	      return *iptr = x + x;
	    INSERT_WORDS64(x,i0&UINT64_C(0x8000000000000000));	/* return +-0 */
	    return x;
	}
}
#ifndef __modf
libm_alias_double (__modf, modf)
#endif
