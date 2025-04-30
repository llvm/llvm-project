/* e_asinf.c -- float version of e_asin.c.
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

/*
  Modifications for single precision expansion are
  Copyright (C) 2001 Stephen L. Moshier <moshier@na-net.ornl.gov>
  and are incorporated herein by permission of the author.  The author
  reserves the right to distribute this material elsewhere under different
  copying permissions.  These modifications are distributed here under
  the following terms:

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, see
    <https://www.gnu.org/licenses/>.  */

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: e_asinf.c,v 1.5 1995/05/12 04:57:25 jtc Exp $";
#endif

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <libm-alias-finite.h>

static const float
one =  1.0000000000e+00, /* 0x3F800000 */
huge =  1.000e+30,

pio2_hi = 1.57079637050628662109375f,
pio2_lo = -4.37113900018624283e-8f,
pio4_hi = 0.785398185253143310546875f,

/* asin x = x + x^3 p(x^2)
   -0.5 <= x <= 0.5;
   Peak relative error 4.8e-9 */
p0 = 1.666675248e-1f,
p1 = 7.495297643e-2f,
p2 = 4.547037598e-2f,
p3 = 2.417951451e-2f,
p4 = 4.216630880e-2f;

float __ieee754_asinf(float x)
{
	float t,w,p,q,c,r,s;
	int32_t hx,ix;
	GET_FLOAT_WORD(hx,x);
	ix = hx&0x7fffffff;
	if(ix==0x3f800000) {
		/* asin(1)=+-pi/2 with inexact */
	    return x*pio2_hi+x*pio2_lo;
	} else if(ix> 0x3f800000) {	/* |x|>= 1 */
	    return (x-x)/(x-x);		/* asin(|x|>1) is NaN */
	} else if (ix<0x3f000000) {	/* |x|<0.5 */
	    if(ix<0x32000000) {		/* if |x| < 2**-27 */
		math_check_force_underflow (x);
		if(huge+x>one) return x;/* return x with inexact if x!=0*/
	    } else {
		t = x*x;
		w = t * (p0 + t * (p1 + t * (p2 + t * (p3 + t * p4))));
		return x+x*w;
	    }
	}
	/* 1> |x|>= 0.5 */
	w = one-fabsf(x);
	t = w*0.5f;
	p = t * (p0 + t * (p1 + t * (p2 + t * (p3 + t * p4))));
	s = sqrtf(t);
	if(ix>=0x3F79999A) {	/* if |x| > 0.975 */
	    t = pio2_hi-(2.0f*(s+s*p)-pio2_lo);
	} else {
	    int32_t iw;
	    w  = s;
	    GET_FLOAT_WORD(iw,w);
	    SET_FLOAT_WORD(w,iw&0xfffff000);
	    c  = (t-w*w)/(s+w);
	    r  = p;
	    p  = 2.0f*s*r-(pio2_lo-2.0f*c);
	    q  = pio4_hi-2.0f*w;
	    t  = pio4_hi-(p-q);
	}
	if(hx>0) return t; else return -t;
}
libm_alias_finite (__ieee754_asinf, __asinf)
