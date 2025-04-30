/* s_nexttowardf.c -- float version of s_nextafter.c.
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com
 * and Jakub Jelinek, jj@ultra.linux.cz.
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

#include <errno.h>
#include <math.h>
#include <math-barriers.h>
#include <math_private.h>
#include <math_ldbl_opt.h>
#include <float.h>

float __nexttowardf(float x, long double y)
{
	int32_t hx,ix;
	int64_t hy,iy;
	double yhi;

	GET_FLOAT_WORD(hx,x);
	yhi = ldbl_high (y);
	EXTRACT_WORDS64 (hy, yhi);
	ix = hx&0x7fffffff;		/* |x| */
	iy = hy&0x7fffffffffffffffLL;	/* |y| */

	if((ix>0x7f800000) ||   /* x is nan */
	   (iy>0x7ff0000000000000LL))
				/* y is nan */
	   return x+y;
	if((long double) x==y) return y;	/* x=y, return y */
	if(ix==0) {				/* x == 0 */
	    float u;
	    SET_FLOAT_WORD(x,(uint32_t)((hy>>32)&0x80000000)|1);/* return +-minsub*/
	    u = math_opt_barrier (x);
	    u = u * u;
	    math_force_eval (u);		/* raise underflow flag */
	    return x;
	}
	if(hx>=0) {				/* x > 0 */
	    if(x > y) {				/* x -= ulp */
		hx -= 1;
	    } else {				/* x < y, x += ulp */
		hx += 1;
	    }
	} else {				/* x < 0 */
	    if(x < y) {				/* x -= ulp */
		hx -= 1;
	    } else {				/* x > y, x += ulp */
		hx += 1;
	    }
	}
	hy = hx&0x7f800000;
	if(hy>=0x7f800000) {
	  float u = x+x;		/* overflow  */
	  math_force_eval (u);
	  __set_errno (ERANGE);
	}
	if(hy<0x00800000) {		/* underflow */
	    float u = x*x;
	    math_force_eval (u);	/* raise underflow flag */
	    __set_errno (ERANGE);
	}
	SET_FLOAT_WORD(x,hx);
	return x;
}
long_double_symbol (libm, __nexttowardf, nexttowardf);
