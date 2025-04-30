/* @(#)s_nextafter.c 5.1 93/09/24 */
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
static char rcsid[] = "$NetBSD: s_nextafter.c,v 1.8 1995/05/10 20:47:58 jtc Exp $";
#endif

/* IEEE functions
 *	nextafter(x,y)
 *	return the next machine floating-point number of x in the
 *	direction toward y.
 *   Special cases:
 */

/* Ugly hack so that the aliasing works.  */
#define __nexttoward __internal___nexttoward
#define nexttoward __internal_nexttoward

#include <errno.h>
#include <math.h>
#include <math-barriers.h>
#include <math_private.h>
#include <float.h>
#include <libm-alias-double.h>

double __nextafter(double x, double y)
{
	int32_t hx,hy,ix,iy;
	uint32_t lx,ly;

	EXTRACT_WORDS(hx,lx,x);
	EXTRACT_WORDS(hy,ly,y);
	ix = hx&0x7fffffff;		/* |x| */
	iy = hy&0x7fffffff;		/* |y| */

	if(((ix>=0x7ff00000)&&((ix-0x7ff00000)|lx)!=0) ||   /* x is nan */
	   ((iy>=0x7ff00000)&&((iy-0x7ff00000)|ly)!=0))     /* y is nan */
	   return x+y;
	if(x==y) return y;		/* x=y, return y */
	if((ix|lx)==0) {			/* x == 0 */
	    double u;
	    INSERT_WORDS(x,hy&0x80000000,1);	/* return +-minsubnormal */
	    u = math_opt_barrier (x);
	    u = u*u;
	    math_force_eval (u);		/* raise underflow flag */
	    return x;
	}
	if(hx>=0) {				/* x > 0 */
	    if(hx>hy||((hx==hy)&&(lx>ly))) {	/* x > y, x -= ulp */
		if(lx==0) hx -= 1;
		lx -= 1;
	    } else {				/* x < y, x += ulp */
		lx += 1;
		if(lx==0) hx += 1;
	    }
	} else {				/* x < 0 */
	    if(hy>=0||hx>hy||((hx==hy)&&(lx>ly))){/* x < y, x -= ulp */
		if(lx==0) hx -= 1;
		lx -= 1;
	    } else {				/* x > y, x += ulp */
		lx += 1;
		if(lx==0) hx += 1;
	    }
	}
	hy = hx&0x7ff00000;
	if(hy>=0x7ff00000) {
	  double u = x+x;	/* overflow  */
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
libm_alias_double (__nextafter, nextafter)
#ifdef NO_LONG_DOUBLE
strong_alias (__nextafter, __nexttowardl)
weak_alias (__nexttowardl, nexttowardl)
#undef __nexttoward
strong_alias (__nextafter, __nexttoward)
#undef nexttoward
weak_alias (__nextafter, nexttoward)
#endif
