/* e_hypotf.c -- float version of e_hypot.c.
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

#include <math.h>
#include <math_private.h>
#include <libm-alias-finite.h>

float
__ieee754_hypotf(float x, float y)
{
	double d_x, d_y;
	int32_t ha, hb;

	GET_FLOAT_WORD(ha,x);
	ha &= 0x7fffffff;
	GET_FLOAT_WORD(hb,y);
	hb &= 0x7fffffff;
	if (ha == 0x7f800000 && !issignaling (y))
	  return fabsf(x);
	else if (hb == 0x7f800000 && !issignaling (x))
	  return fabsf(y);
	else if (ha > 0x7f800000 || hb > 0x7f800000)
	  return fabsf(x) * fabsf(y);
	else if (ha == 0)
	  return fabsf(y);
	else if (hb == 0)
	  return fabsf(x);

	d_x = (double) x;
	d_y = (double) y;

	return (float) sqrt(d_x * d_x + d_y * d_y);
}
#ifndef __ieee754_hypotf
libm_alias_finite (__ieee754_hypotf, __hypotf)
#endif
