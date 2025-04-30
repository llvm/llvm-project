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
 * scalbn (double x, int n)
 * scalbn(x,n) returns x* 2**n  computed by  exponent
 * manipulation rather than by actually performing an
 * exponentiation or a multiplication.
 */

#include <math.h>
#include <math_private.h>

static const double
two54   =  1.80143985094819840000e+16, /* 0x43500000, 0x00000000 */
twom54  =  5.55111512312578270212e-17, /* 0x3C900000, 0x00000000 */
huge   = 1.0e+300,
tiny   = 1.0e-300;

double
__scalbn (double x, int n)
{
	int64_t ix;
	int64_t k;
	EXTRACT_WORDS64(ix,x);
	k = (ix >> 52) & 0x7ff;			/* extract exponent */
	if (__builtin_expect(k==0, 0)) {	/* 0 or subnormal x */
	    if ((ix & UINT64_C(0xfffffffffffff))==0) return x; /* +-0 */
	    x *= two54;
	    EXTRACT_WORDS64(ix,x);
	    k = ((ix >> 52) & 0x7ff) - 54;
	    }
	if (__builtin_expect(k==0x7ff, 0)) return x+x;	/* NaN or Inf */
	if (__builtin_expect(n< -50000, 0))
	  return tiny*copysign(tiny,x); /*underflow*/
	if (__builtin_expect(n> 50000 || k+n > 0x7fe, 0))
	  return huge*copysign(huge,x); /* overflow  */
	/* Now k and n are bounded we know that k = k+n does not
	   overflow.  */
	k = k+n;
	if (__builtin_expect(k > 0, 1))		/* normal result */
	    {INSERT_WORDS64(x,(ix&UINT64_C(0x800fffffffffffff))|(k<<52));
	      return x;}
	if (k <= -54)
	  return tiny*copysign(tiny,x);	/*underflow*/
	k += 54;				/* subnormal result */
	INSERT_WORDS64(x,(ix&INT64_C(0x800fffffffffffff))|(k<<52));
	return x*twom54;
}
#ifdef NO_LONG_DOUBLE
strong_alias (__scalbn, __scalbnl)
#endif
