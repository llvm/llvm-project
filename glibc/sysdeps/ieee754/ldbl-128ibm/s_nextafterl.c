/* s_nextafterl.c -- long double version of s_nextafter.c.
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

/* IEEE functions
 *	nextafterl(x,y)
 *	return the next machine floating-point number of x in the
 *	direction toward y.
 *   Special cases:
 */

#include <errno.h>
#include <float.h>
#include <math.h>
#include <math-barriers.h>
#include <math_private.h>
#include <math_ldbl_opt.h>

long double __nextafterl(long double x, long double y)
{
	int64_t hx, hy, ihx, ihy, lx;
	double xhi, xlo, yhi;

	ldbl_unpack (x, &xhi, &xlo);
	EXTRACT_WORDS64 (hx, xhi);
	EXTRACT_WORDS64 (lx, xlo);
	yhi = ldbl_high (y);
	EXTRACT_WORDS64 (hy, yhi);
	ihx = hx&0x7fffffffffffffffLL;		/* |hx| */
	ihy = hy&0x7fffffffffffffffLL;		/* |hy| */

	if((ihx>0x7ff0000000000000LL) ||	/* x is nan */
	   (ihy>0x7ff0000000000000LL))		/* y is nan */
	    return x+y; /* signal the nan */
	if(x==y)
	    return y;		/* x=y, return y */
	if(ihx == 0) {				/* x == 0 */
	    long double u;			/* return +-minsubnormal */
	    hy = (hy & 0x8000000000000000ULL) | 1;
	    INSERT_WORDS64 (yhi, hy);
	    x = yhi;
	    u = math_opt_barrier (x);
	    u = u * u;
	    math_force_eval (u);		/* raise underflow flag */
	    return x;
	}

	long double u;
	if(x > y) {	/* x > y, x -= ulp */
	    /* This isn't the largest magnitude correctly rounded
	       long double as you can see from the lowest mantissa
	       bit being zero.  It is however the largest magnitude
	       long double with a 106 bit mantissa, and nextafterl
	       is insane with variable precision.  So to make
	       nextafterl sane we assume 106 bit precision.  */
	    if((hx==0xffefffffffffffffLL)&&(lx==0xfc8ffffffffffffeLL)) {
	      u = x+x;	/* overflow, return -inf */
	      math_force_eval (u);
	      __set_errno (ERANGE);
	      return y;
	    }
	    if (hx >= 0x7ff0000000000000LL) {
	      u = 0x1.fffffffffffff7ffffffffffff8p+1023L;
	      return u;
	    }
	    if(ihx <= 0x0360000000000000LL) {  /* x <= LDBL_MIN */
	      u = math_opt_barrier (x);
	      x -= LDBL_TRUE_MIN;
	      if (ihx < 0x0360000000000000LL
		  || (hx > 0 && lx <= 0)
		  || (hx < 0 && lx > 1)) {
		u = u * u;
		math_force_eval (u);		/* raise underflow flag */
		__set_errno (ERANGE);
	      }
	      /* Avoid returning -0 in FE_DOWNWARD mode.  */
	      if (x == 0.0L)
		return 0.0L;
	      return x;
	    }
	    /* If the high double is an exact power of two and the low
	       double is the opposite sign, then 1ulp is one less than
	       what we might determine from the high double.  Similarly
	       if X is an exact power of two, and positive, because
	       making it a little smaller will result in the exponent
	       decreasing by one and normalisation of the mantissa.   */
	    if ((hx & 0x000fffffffffffffLL) == 0
		&& ((lx != 0 && (hx ^ lx) < 0)
		    || (lx == 0 && hx >= 0)))
	      ihx -= 1LL << 52;
	    if (ihx < (106LL << 52)) { /* ulp will denormal */
	      INSERT_WORDS64 (yhi, ihx & (0x7ffLL<<52));
	      u = yhi * 0x1p-105;
	    } else {
	      INSERT_WORDS64 (yhi, (ihx & (0x7ffLL<<52))-(105LL<<52));
	      u = yhi;
	    }
	    return x - u;
	} else {				/* x < y, x += ulp */
	    if((hx==0x7fefffffffffffffLL)&&(lx==0x7c8ffffffffffffeLL)) {
	      u = x+x;	/* overflow, return +inf */
	      math_force_eval (u);
	      __set_errno (ERANGE);
	      return y;
	    }
	    if ((uint64_t) hx >= 0xfff0000000000000ULL) {
	      u = -0x1.fffffffffffff7ffffffffffff8p+1023L;
	      return u;
	    }
	    if(ihx <= 0x0360000000000000LL) {  /* x <= LDBL_MIN */
	      u = math_opt_barrier (x);
	      x += LDBL_TRUE_MIN;
	      if (ihx < 0x0360000000000000LL
		  || (hx > 0 && lx < 0 && lx != 0x8000000000000001LL)
		  || (hx < 0 && lx >= 0)) {
		u = u * u;
		math_force_eval (u);		/* raise underflow flag */
		__set_errno (ERANGE);
	      }
	      if (x == 0.0L)	/* handle negative LDBL_TRUE_MIN case */
		x = -0.0L;
	      return x;
	    }
	    /* If the high double is an exact power of two and the low
	       double is the opposite sign, then 1ulp is one less than
	       what we might determine from the high double.  Similarly
	       if X is an exact power of two, and negative, because
	       making it a little larger will result in the exponent
	       decreasing by one and normalisation of the mantissa.   */
	    if ((hx & 0x000fffffffffffffLL) == 0
		&& ((lx != 0 && (hx ^ lx) < 0)
		    || (lx == 0 && hx < 0)))
	      ihx -= 1LL << 52;
	    if (ihx < (106LL << 52)) { /* ulp will denormal */
	      INSERT_WORDS64 (yhi, ihx & (0x7ffLL<<52));
	      u = yhi * 0x1p-105;
	    } else {
	      INSERT_WORDS64 (yhi, (ihx & (0x7ffLL<<52))-(105LL<<52));
	      u = yhi;
	    }
	    return x + u;
	}
}
strong_alias (__nextafterl, __nexttowardl)
long_double_symbol (libm, __nextafterl, nextafterl);
long_double_symbol (libm, __nexttowardl, nexttowardl);
