/* e_fmodl.c -- long double version of e_fmod.c.
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

/* __ieee754_remainderl(x,p)
 * Return :
 *	returns  x REM p  =  x - [x/p]*p as if in infinite
 *	precise arithmetic, where [x/p] is the (infinite bit)
 *	integer nearest x/p (in half way case choose the even one).
 * Method :
 *	Based on fmodl() return x-[x/p]chopped*p exactlp.
 */

#include <math.h>
#include <math_private.h>
#include <libm-alias-finite.h>

static const long double zero = 0.0L;


long double
__ieee754_remainderl(long double x, long double p)
{
	int64_t hx,hp;
	uint64_t sx,lx,lp;
	long double p_half;
	double xhi, xlo, phi, plo;

	ldbl_unpack (x, &xhi, &xlo);
	EXTRACT_WORDS64 (hx, xhi);
	EXTRACT_WORDS64 (lx, xlo);
	ldbl_unpack (p, &phi, &plo);
	EXTRACT_WORDS64 (hp, phi);
	EXTRACT_WORDS64 (lp, plo);
	sx = hx&0x8000000000000000ULL;
	lp ^= hp & 0x8000000000000000ULL;
	hp &= 0x7fffffffffffffffLL;
	lx ^= sx;
	hx &= 0x7fffffffffffffffLL;
	if (lp == 0x8000000000000000ULL)
	  lp = 0;
	if (lx == 0x8000000000000000ULL)
	  lx = 0;

    /* purge off exception values */
	if(hp==0) return (x*p)/(x*p);	/* p = 0 */
	if((hx>=0x7ff0000000000000LL)||			/* x not finite */
	   (hp>0x7ff0000000000000LL))			/* p is NaN */
	    return (x*p)/(x*p);


	if (hp<=0x7fdfffffffffffffLL) x = __ieee754_fmodl(x,p+p);	/* now x < 2p */
	if (((hx-hp)|(lx-lp))==0) return zero*x;
	x  = fabsl(x);
	p  = fabsl(p);
	if (hp<0x0020000000000000LL) {
	    if(x+x>p) {
		x-=p;
		if(x+x>=p) x -= p;
	    }
	} else {
	    p_half = 0.5L*p;
	    if(x>p_half) {
		x-=p;
		if(x>=p_half) x -= p;
	    }
	}
	if (sx)
	  x = -x;
	return x;
}
libm_alias_finite (__ieee754_remainderl, __remainderl)
