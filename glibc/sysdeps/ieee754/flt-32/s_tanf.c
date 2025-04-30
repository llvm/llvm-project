/* s_tanf.c -- float version of s_tan.c.
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

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: s_tanf.c,v 1.4 1995/05/10 20:48:20 jtc Exp $";
#endif

#include <errno.h>
#include <math.h>
#include <math_private.h>
#include <libm-alias-float.h>
#include "s_sincosf.h"

/* Reduce range of X to a multiple of PI/2.  The modulo result is between
   -PI/4 and PI/4 and returned as a high part y[0] and a low part y[1].
   The low bit in the return value indicates the first or 2nd half of tanf.  */
static inline int32_t
rem_pio2f (float x, float *y)
{
  double dx = x;
  int n;
  const sincos_t *p = &__sincosf_table[0];

  if (__glibc_likely (abstop12 (x) < abstop12 (120.0f)))
    dx = reduce_fast (dx, p, &n);
  else
    {
      uint32_t xi = asuint (x);
      int sign = xi >> 31;

      dx = reduce_large (xi, &n);
      dx = sign ? -dx : dx;
    }

  y[0] = dx;
  y[1] = dx - y[0];
  return n;
}

float __tanf(float x)
{
	float y[2],z=0.0;
	int32_t n, ix;

	GET_FLOAT_WORD(ix,x);

    /* |x| ~< pi/4 */
	ix &= 0x7fffffff;
	if(ix <= 0x3f490fda) return __kernel_tanf(x,z,1);

    /* tan(Inf or NaN) is NaN */
	else if (ix>=0x7f800000) {
	  if (ix==0x7f800000)
	    __set_errno (EDOM);
	  return x-x;		/* NaN */
	}

    /* argument reduction needed */
	else {
	    n = rem_pio2f(x,y);
	    return __kernel_tanf(y[0],y[1],1-((n&1)<<1)); /*   1 -- n even
							      -1 -- n odd */
	}
}
libm_alias_float (__tan, tan)
