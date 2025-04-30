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
  Long double expansions are
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

/* __kernel_tanl( x, y, k )
 * kernel tan function on [-pi/4, pi/4], pi/4 ~ 0.7854
 * Input x is assumed to be bounded by ~pi/4 in magnitude.
 * Input y is the tail of x.
 * Input k indicates whether tan (if k=1) or
 * -1/tan (if k= -1) is returned.
 *
 * Algorithm
 *	1. Since tan(-x) = -tan(x), we need only to consider positive x.
 *	2. if x < 2^-57, return x with inexact if x!=0.
 *	3. tan(x) is approximated by a rational form x + x^3 / 3 + x^5 R(x^2)
 *          on [0,0.67433].
 *
 *	   Note: tan(x+y) = tan(x) + tan'(x)*y
 *		          ~ tan(x) + (1+x*x)*y
 *	   Therefore, for better accuracy in computing tan(x+y), let
 *		r = x^3 * R(x^2)
 *	   then
 *		tan(x+y) = x + (x^3 / 3 + (x^2 *(r+y)+y))
 *
 *      4. For x in [0.67433,pi/4],  let y = pi/4 - x, then
 *		tan(x) = tan(pi/4-y) = (1-tan(y))/(1+tan(y))
 *		       = 1 - 2*(tan(y) - (tan(y)^2)/(1+tan(y)))
 */

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <libc-diag.h>

static const _Float128
  one = 1,
  pio4hi = L(7.8539816339744830961566084581987569936977E-1),
  pio4lo = L(2.1679525325309452561992610065108379921906E-35),

  /* tan x = x + x^3 / 3 + x^5 T(x^2)/U(x^2)
     0 <= x <= 0.6743316650390625
     Peak relative error 8.0e-36  */
 TH =  L(3.333333333333333333333333333333333333333E-1),
 T0 = L(-1.813014711743583437742363284336855889393E7),
 T1 =  L(1.320767960008972224312740075083259247618E6),
 T2 = L(-2.626775478255838182468651821863299023956E4),
 T3 =  L(1.764573356488504935415411383687150199315E2),
 T4 = L(-3.333267763822178690794678978979803526092E-1),

 U0 = L(-1.359761033807687578306772463253710042010E8),
 U1 =  L(6.494370630656893175666729313065113194784E7),
 U2 = L(-4.180787672237927475505536849168729386782E6),
 U3 =  L(8.031643765106170040139966622980914621521E4),
 U4 = L(-5.323131271912475695157127875560667378597E2);
  /* 1.000000000000000000000000000000000000000E0 */


_Float128
__kernel_tanl (_Float128 x, _Float128 y, int iy)
{
  _Float128 z, r, v, w, s;
  int32_t ix, sign;
  ieee854_long_double_shape_type u, u1;

  u.value = x;
  ix = u.parts32.w0 & 0x7fffffff;
  if (ix < 0x3fc60000)		/* x < 2**-57 */
    {
      if ((int) x == 0)
	{			/* generate inexact */
	  if ((ix | u.parts32.w1 | u.parts32.w2 | u.parts32.w3
	       | (iy + 1)) == 0)
	    return one / fabsl (x);
	  else if (iy == 1)
	    {
	      math_check_force_underflow (x);
	      return x;
	    }
	  else
	    return -one / x;
	}
    }
  if (ix >= 0x3ffe5942) /* |x| >= 0.6743316650390625 */
    {
      if ((u.parts32.w0 & 0x80000000) != 0)
	{
	  x = -x;
	  y = -y;
	  sign = -1;
	}
      else
	sign = 1;
      z = pio4hi - x;
      w = pio4lo - y;
      x = z + w;
      y = 0.0;
    }
  z = x * x;
  r = T0 + z * (T1 + z * (T2 + z * (T3 + z * T4)));
  v = U0 + z * (U1 + z * (U2 + z * (U3 + z * (U4 + z))));
  r = r / v;

  s = z * x;
  r = y + z * (s * r + y);
  r += TH * s;
  w = x + r;
  if (ix >= 0x3ffe5942)
    {
      v = (_Float128) iy;
      w = (v - 2.0 * (x - (w * w / (w + v) - r)));
      /* SIGN is set for arguments that reach this code, but not
	 otherwise, resulting in warnings that it may be used
	 uninitialized although in the cases where it is used it has
	 always been set.  */
      DIAG_PUSH_NEEDS_COMMENT;
#if defined(__clang__)
      DIAG_IGNORE_NEEDS_COMMENT (4.7, "-Wsometimes-uninitialized");
#else
      DIAG_IGNORE_NEEDS_COMMENT (4.7, "-Wmaybe-uninitialized");
#endif
      if (sign < 0)
	w = -w;
      DIAG_POP_NEEDS_COMMENT;
      return w;
    }
  if (iy == 1)
    return w;
  else
    {				/* if allow error up to 2 ulp,
				   simply return -1.0/(x+r) here */
      /*  compute -1.0/(x+r) accurately */
      u1.value = w;
      u1.parts32.w2 = 0;
      u1.parts32.w3 = 0;
      v = r - (u1.value - x);		/* u1+v = r+x */
      z = -1.0 / w;
      u.value = z;
      u.parts32.w2 = 0;
      u.parts32.w3 = 0;
      s = 1.0 + u.value * u1.value;
      return u.value + z * (s + u.value * v);
    }
}
