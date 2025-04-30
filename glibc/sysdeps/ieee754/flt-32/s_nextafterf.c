/* s_nextafterf.c -- float version of s_nextafter.c.
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
static char rcsid[] = "$NetBSD: s_nextafterf.c,v 1.4 1995/05/10 20:48:01 jtc Exp $";
#endif

#include <errno.h>
#include <math.h>
#include <math-barriers.h>
#include <math_private.h>
#include <libm-alias-float.h>
#include <float.h>

float __nextafterf(float x, float y)
{
	int32_t hx,hy,ix,iy;

	GET_FLOAT_WORD(hx,x);
	GET_FLOAT_WORD(hy,y);
	ix = hx&0x7fffffff;		/* |x| */
	iy = hy&0x7fffffff;		/* |y| */

	if((ix>0x7f800000) ||   /* x is nan */
	   (iy>0x7f800000))     /* y is nan */
	   return x+y;
	if(x==y) return y;		/* x=y, return y */
	if(ix==0) {				/* x == 0 */
	    float u;
	    SET_FLOAT_WORD(x,(hy&0x80000000)|1);/* return +-minsubnormal */
	    u = math_opt_barrier (x);
	    u = u*u;
	    math_force_eval (u);		/* raise underflow flag */
	    return x;
	}
	if(hx>=0) {				/* x > 0 */
	    if(hx>hy) {				/* x > y, x -= ulp */
		hx -= 1;
	    } else {				/* x < y, x += ulp */
		hx += 1;
	    }
	} else {				/* x < 0 */
	    if(hy>=0||hx>hy){			/* x < y, x -= ulp */
		hx -= 1;
	    } else {				/* x > y, x += ulp */
		hx += 1;
	    }
	}
	hy = hx&0x7f800000;
	if(hy>=0x7f800000) {
	  float u = x+x;	/* overflow  */
	  math_force_eval (u);
	  __set_errno (ERANGE);
	}
	if(hy<0x00800000) {
	    float u = x*x;			/* underflow */
	    math_force_eval (u);		/* raise underflow flag */
	    __set_errno (ERANGE);
	}
	SET_FLOAT_WORD(x,hx);
	return x;
}
libm_alias_float (__nextafter, nextafter)
