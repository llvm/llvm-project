/* e_atan2l.c -- long double version of e_atan2.c.
 * Conversion to long double by Jakub Jelinek, jj@ultra.linux.cz.
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

/* __ieee754_atan2l(y,x)
 * Method :
 *	1. Reduce y to positive by atan2l(y,x)=-atan2l(-y,x).
 *	2. Reduce x to positive by (if x and y are unexceptional):
 *		ARG (x+iy) = arctan(y/x)	   ... if x > 0,
 *		ARG (x+iy) = pi - arctan[y/(-x)]   ... if x < 0,
 *
 * Special cases:
 *
 *	ATAN2((anything), NaN ) is NaN;
 *	ATAN2(NAN , (anything) ) is NaN;
 *	ATAN2(+-0, +(anything but NaN)) is +-0  ;
 *	ATAN2(+-0, -(anything but NaN)) is +-pi ;
 *	ATAN2(+-(anything but 0 and NaN), 0) is +-pi/2;
 *	ATAN2(+-(anything but INF and NaN), +INF) is +-0 ;
 *	ATAN2(+-(anything but INF and NaN), -INF) is +-pi;
 *	ATAN2(+-INF,+INF ) is +-pi/4 ;
 *	ATAN2(+-INF,-INF ) is +-3pi/4;
 *	ATAN2(+-INF, (anything but,0,NaN, and INF)) is +-pi/2;
 *
 * Constants:
 * The hexadecimal values are the intended ones for the following
 * constants. The decimal values may be used, provided that the
 * compiler will convert from decimal to binary accurately enough
 * to produce the hexadecimal values shown.
 */

#include <math.h>
#include <math_private.h>
#include <libm-alias-finite.h>

static const long double
tiny  = 1.0e-300L,
zero  = 0.0,
pi_o_4  = 7.85398163397448309615660845819875699e-01L, /* 3ffe921fb54442d18469898cc51701b8 */
pi_o_2  = 1.57079632679489661923132169163975140e+00L, /* 3fff921fb54442d18469898cc51701b8 */
pi      = 3.14159265358979323846264338327950280e+00L, /* 4000921fb54442d18469898cc51701b8 */
pi_lo   = 8.67181013012378102479704402604335225e-35L; /* 3f8dcd129024e088a67cc74020bbea64 */

long double
__ieee754_atan2l(long double y, long double x)
{
	long double z;
	int64_t k,m,hx,hy,ix,iy;
	uint64_t lx;
	double xhi, xlo, yhi;

	ldbl_unpack (x, &xhi, &xlo);
	EXTRACT_WORDS64 (hx, xhi);
	EXTRACT_WORDS64 (lx, xlo);
	ix = hx&0x7fffffffffffffffLL;
	yhi = ldbl_high (y);
	EXTRACT_WORDS64 (hy, yhi);
	iy = hy&0x7fffffffffffffffLL;
	if(((ix)>0x7ff0000000000000LL)||
	   ((iy)>0x7ff0000000000000LL))	/* x or y is NaN */
	   return x+y;
	if(((hx-0x3ff0000000000000LL))==0
	   && (lx&0x7fffffffffffffff)==0) return __atanl(y);   /* x=1.0L */
	m = ((hy>>63)&1)|((hx>>62)&2);	/* 2*sign(x)+sign(y) */

    /* when y = 0 */
	if(iy==0) {
	    switch(m) {
		case 0:
		case 1: return y;	/* atan(+-0,+anything)=+-0 */
		case 2: return  pi+tiny;/* atan(+0,-anything) = pi */
		case 3: return -pi-tiny;/* atan(-0,-anything) =-pi */
	    }
	}
    /* when x = 0 */
	if(ix==0) return (hy<0)?  -pi_o_2-tiny: pi_o_2+tiny;

    /* when x is INF */
	if(ix==0x7ff0000000000000LL) {
	    if(iy==0x7ff0000000000000LL) {
		switch(m) {
		    case 0: return  pi_o_4+tiny;/* atan(+INF,+INF) */
		    case 1: return -pi_o_4-tiny;/* atan(-INF,+INF) */
		    case 2: return  3.0L*pi_o_4+tiny;/*atan(+INF,-INF)*/
		    case 3: return -3.0L*pi_o_4-tiny;/*atan(-INF,-INF)*/
		}
	    } else {
		switch(m) {
		    case 0: return  zero  ;	/* atan(+...,+INF) */
		    case 1: return -zero  ;	/* atan(-...,+INF) */
		    case 2: return  pi+tiny  ;	/* atan(+...,-INF) */
		    case 3: return -pi-tiny  ;	/* atan(-...,-INF) */
		}
	    }
	}
    /* when y is INF */
	if(iy==0x7ff0000000000000LL) return (hy<0)? -pi_o_2-tiny: pi_o_2+tiny;

    /* compute y/x */
	k = (iy-ix)>>52;
	if(k > 120) z=pi_o_2+0.5L*pi_lo;	/* |y/x| >  2**120 */
	else if(hx<0&&k<-120) z=0.0L;		/* |y|/x < -2**120 */
	else z=__atanl(fabsl(y/x));		/* safe to do y/x */
	switch (m) {
	    case 0: return       z  ;	/* atan(+,+) */
	    case 1: return      -z  ;	/* atan(-,+) */
	    case 2: return  pi-(z-pi_lo);/* atan(+,-) */
	    default: /* case 3 */
		    return  (z-pi_lo)-pi;/* atan(-,-) */
	}
}
libm_alias_finite (__ieee754_atan2l, __atan2l)
