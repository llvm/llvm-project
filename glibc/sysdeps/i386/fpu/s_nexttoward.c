/* s_nexttoward.c
 * Special i387 version
 * Conversion from s_nextafter.c by Ulrich Drepper, Cygnus Support,
 * drepper@cygnus.com.
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

/* IEEE functions
 *	nexttoward(x,y)
 *	return the next machine floating-point number of x in the
 *	direction toward y.
 *   Special cases:
 */

#include <errno.h>
#include <math.h>
#include <math-barriers.h>
#include <math_private.h>
#include <float.h>

double __nexttoward(double x, long double y)
{
	int32_t hx,ix,iy;
	uint32_t lx,hy,ly,esy;

	EXTRACT_WORDS(hx,lx,x);
	GET_LDOUBLE_WORDS(esy,hy,ly,y);
	ix = hx&0x7fffffff;		/* |x| */
	iy = esy&0x7fff;		/* |y| */

	/* Intel's extended format has the normally implicit 1 explicit
	   present.  Sigh!  */
	if(((ix>=0x7ff00000)&&((ix-0x7ff00000)|lx)!=0) ||   /* x is nan */
	   ((iy>=0x7fff)&&((hy&0x7fffffff)|ly)!=0))        /* y is nan */
	   return x+y;
	if((long double) x==y) return y;	/* x=y, return y */
	if((ix|lx)==0) {			/* x == 0 */
	    double u;
	    INSERT_WORDS(x,(esy&0x8000)<<16,1); /* return +-minsub */
	    u = math_opt_barrier (x);
	    u = u * u;
	    math_force_eval (u);		/* raise underflow flag */
	    return x;
	}
	if(hx>=0) {				/* x > 0 */
	    if (x > y) {			/* x -= ulp */
		if(lx==0) hx -= 1;
		lx -= 1;
	    } else {				/* x < y, x += ulp */
		lx += 1;
		if(lx==0) hx += 1;
	    }
	} else {				/* x < 0 */
	    if (x < y) {			/* x -= ulp */
		if(lx==0) hx -= 1;
		lx -= 1;
	    } else {				/* x > y, x += ulp */
		lx += 1;
		if(lx==0) hx += 1;
	    }
	}
	hy = hx&0x7ff00000;
	if(hy>=0x7ff00000) {
	  double u = x+x;			/* overflow  */
	  math_force_eval (u);
	  __set_errno (ERANGE);
	}
	if(hy<0x00100000) {
	    double u = x*x;			/* underflow */
	    math_force_eval (u);		/* raise underflow flag */
	    __set_errno (ERANGE);
	}
	INSERT_WORDS(x,hx,lx);
	return x;
}
weak_alias (__nexttoward, nexttoward)
#ifdef NO_LONG_DOUBLE
strong_alias (__nexttoward, __nexttowardl)
weak_alias (__nexttoward, nexttowardl)
#endif
