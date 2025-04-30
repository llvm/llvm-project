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
 * __ieee754_fmod(x,y)
 * Return x mod y in exact arithmetic
 * Method: shift and subtract
 */

#include <math.h>
#include <math_private.h>
#include <stdint.h>
#include <libm-alias-finite.h>

static const double one = 1.0, Zero[] = {0.0, -0.0,};

double
__ieee754_fmod (double x, double y)
{
	int32_t n,ix,iy;
	int64_t hx,hy,hz,sx,i;

	EXTRACT_WORDS64(hx,x);
	EXTRACT_WORDS64(hy,y);
	sx = hx&UINT64_C(0x8000000000000000);	/* sign of x */
	hx ^=sx;				/* |x| */
	hy &= UINT64_C(0x7fffffffffffffff);	/* |y| */

    /* purge off exception values */
	if(__builtin_expect(hy==0
			    || hx >= UINT64_C(0x7ff0000000000000)
			    || hy > UINT64_C(0x7ff0000000000000), 0))
	  /* y=0,or x not finite or y is NaN */
	    return (x*y)/(x*y);
	if(__builtin_expect(hx<=hy, 0)) {
	    if(hx<hy) return x;	/* |x|<|y| return x */
	    return Zero[(uint64_t)sx>>63];	/* |x|=|y| return x*0*/
	}

    /* determine ix = ilogb(x) */
	if(__builtin_expect(hx<UINT64_C(0x0010000000000000), 0)) {
	  /* subnormal x */
	  for (ix = -1022,i=(hx<<11); i>0; i<<=1) ix -=1;
	} else ix = (hx>>52)-1023;

    /* determine iy = ilogb(y) */
	if(__builtin_expect(hy<UINT64_C(0x0010000000000000), 0)) {	/* subnormal y */
	  for (iy = -1022,i=(hy<<11); i>0; i<<=1) iy -=1;
	} else iy = (hy>>52)-1023;

    /* set up hx, hy and align y to x */
	if(__builtin_expect(ix >= -1022, 1))
	    hx = UINT64_C(0x0010000000000000)|(UINT64_C(0x000fffffffffffff)&hx);
	else {		/* subnormal x, shift x to normal */
	    n = -1022-ix;
	    hx<<=n;
	}
	if(__builtin_expect(iy >= -1022, 1))
	    hy = UINT64_C(0x0010000000000000)|(UINT64_C(0x000fffffffffffff)&hy);
	else {		/* subnormal y, shift y to normal */
	    n = -1022-iy;
	    hy<<=n;
	}

    /* fix point fmod */
	n = ix - iy;
	while(n--) {
	    hz=hx-hy;
	    if(hz<0){hx = hx+hx;}
	    else {
		if(hz==0)		/* return sign(x)*0 */
		    return Zero[(uint64_t)sx>>63];
		hx = hz+hz;
	    }
	}
	hz=hx-hy;
	if(hz>=0) {hx=hz;}

    /* convert back to floating value and restore the sign */
	if(hx==0)			/* return sign(x)*0 */
	    return Zero[(uint64_t)sx>>63];
	while(hx<UINT64_C(0x0010000000000000)) {	/* normalize x */
	    hx = hx+hx;
	    iy -= 1;
	}
	if(__builtin_expect(iy>= -1022, 1)) {	/* normalize output */
	  hx = ((hx-UINT64_C(0x0010000000000000))|((uint64_t)(iy+1023)<<52));
	    INSERT_WORDS64(x,hx|sx);
	} else {		/* subnormal output */
	    n = -1022 - iy;
	    hx>>=n;
	    INSERT_WORDS64(x,hx|sx);
	    x *= one;		/* create necessary signal */
	}
	return x;		/* exact output */
}
libm_alias_finite (__ieee754_fmod, __fmod)
