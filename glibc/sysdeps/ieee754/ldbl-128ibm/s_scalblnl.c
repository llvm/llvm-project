/* s_scalblnl.c -- long double version of s_scalbln.c.
 * Conversion to IEEE quad long double by Jakub Jelinek, jj@ultra.linux.cz.
 */

/* @(#)s_scalbln.c 5.1 93/09/24 */
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
 * scalblnl (long double x, long int n)
 * scalblnl(x,n) returns x* 2**n  computed by  exponent
 * manipulation rather than by actually performing an
 * exponentiation or a multiplication.
 */

#include <math.h>
#include <math_private.h>
#include <math_ldbl_opt.h>

static const long double
twolm54 = 5.55111512312578270212e-17, /* 0x3C90000000000000, 0 */
huge   = 1.0E+300L,
tiny   = 1.0E-300L;
static const double
two54 = 1.80143985094819840000e+16, /* 0x4350000000000000 */
twom54 = 5.55111512312578270212e-17; /* 0x3C90000000000000 */

long double __scalblnl (long double x, long int n)
{
	int64_t k,l,hx,lx;
	union { int64_t i; double d; } u;
	double xhi, xlo;

	ldbl_unpack (x, &xhi, &xlo);
	EXTRACT_WORDS64 (hx, xhi);
	EXTRACT_WORDS64 (lx, xlo);
	k = (hx>>52)&0x7ff;		/* extract exponent */
	l = (lx>>52)&0x7ff;
	if (k==0) {				/* 0 or subnormal x */
	    if ((hx&0x7fffffffffffffffULL)==0) return x; /* +-0 */
	    u.i = hx;
	    u.d *= two54;
	    hx = u.i;
	    k = ((hx>>52)&0x7ff) - 54;
	}
	else if (k==0x7ff) return x+x;		/* NaN or Inf */
	if (n< -50000) return tiny*copysignl(tiny,x); /*underflow */
	if (n> 50000 || k+n > 0x7fe)
	  return huge*copysignl(huge,x); /* overflow */
	/* Now k and n are bounded we know that k = k+n does not
	   overflow.  */
	k = k+n;
	if (k > 0) {				/* normal result */
	    hx = (hx&0x800fffffffffffffULL)|(k<<52);
	    if ((lx & 0x7fffffffffffffffULL) == 0) { /* low part +-0 */
		INSERT_WORDS64 (xhi, hx);
		INSERT_WORDS64 (xlo, lx);
		x = ldbl_pack (xhi, xlo);
		return x;
	    }
	    if (l == 0) { /* low part subnormal */
		u.i = lx;
		u.d *= two54;
		lx = u.i;
		l = ((lx>>52)&0x7ff) - 54;
	    }
	    l = l + n;
	    if (l > 0)
		lx = (lx&0x800fffffffffffffULL)|(l<<52);
	    else if (l <= -54)
		lx = (lx&0x8000000000000000ULL);
	    else {
		l += 54;
		u.i = (lx&0x800fffffffffffffULL)|(l<<52);
		u.d *= twom54;
		lx = u.i;
	    }
	    INSERT_WORDS64 (xhi, hx);
	    INSERT_WORDS64 (xlo, lx);
	    x = ldbl_pack (xhi, xlo);
	    return x;
	}
	if (k <= -54)
	  return tiny*copysignl(tiny,x); 	/*underflow*/
	k += 54;				/* subnormal result */
	lx &= 0x8000000000000000ULL;
	hx &= 0x800fffffffffffffULL;
	INSERT_WORDS64 (xhi, hx|(k<<52));
	INSERT_WORDS64 (xlo, lx);
	x = ldbl_pack (xhi, xlo);
	return x*twolm54;
}
