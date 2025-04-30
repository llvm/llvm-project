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
  copying permissions.  These modifications are distributed here under the
  following terms:

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

/* __ieee754_asin(x)
 * Method :
 *	Since  asin(x) = x + x^3/6 + x^5*3/40 + x^7*15/336 + ...
 *	we approximate asin(x) on [0,0.5] by
 *		asin(x) = x + x*x^2*R(x^2)
 *      Between .5 and .625 the approximation is
 *              asin(0.5625 + x) = asin(0.5625) + x rS(x) / sS(x)
 *	For x in [0.625,1]
 *		asin(x) = pi/2-2*asin(sqrt((1-x)/2))
 *	Let y = (1-x), z = y/2, s := sqrt(z), and pio2_hi+pio2_lo=pi/2;
 *	then for x>0.98
 *		asin(x) = pi/2 - 2*(s+s*z*R(z))
 *			= pio2_hi - (2*(s+s*z*R(z)) - pio2_lo)
 *	For x<=0.98, let pio4_hi = pio2_hi/2, then
 *		f = hi part of s;
 *		c = sqrt(z) - f = (z-f*f)/(s+f) 	...f+c=sqrt(z)
 *	and
 *		asin(x) = pi/2 - 2*(s+s*z*R(z))
 *			= pio4_hi+(pio4-2s)-(2s*z*R(z)-pio2_lo)
 *			= pio4_hi+(pio4-2f)-(2s*z*R(z)-(pio2_lo+2c))
 *
 * Special cases:
 *	if x is NaN, return x itself;
 *	if |x|>1, return NaN with invalid signal.
 *
 */


#include <float.h>
#include <math.h>
#include <math-barriers.h>
#include <math_private.h>
#include <math-underflow.h>
#include <libm-alias-finite.h>

static const _Float128
  one = 1,
  huge = L(1.0e+4932),
  pio2_hi = L(1.5707963267948966192313216916397514420986),
  pio2_lo = L(4.3359050650618905123985220130216759843812E-35),
  pio4_hi = L(7.8539816339744830961566084581987569936977E-1),

	/* coefficient for R(x^2) */

  /* asin(x) = x + x^3 pS(x^2) / qS(x^2)
     0 <= x <= 0.5
     peak relative error 1.9e-35  */
  pS0 = L(-8.358099012470680544198472400254596543711E2),
  pS1 =  L(3.674973957689619490312782828051860366493E3),
  pS2 = L(-6.730729094812979665807581609853656623219E3),
  pS3 =  L(6.643843795209060298375552684423454077633E3),
  pS4 = L(-3.817341990928606692235481812252049415993E3),
  pS5 =  L(1.284635388402653715636722822195716476156E3),
  pS6 = L(-2.410736125231549204856567737329112037867E2),
  pS7 =  L(2.219191969382402856557594215833622156220E1),
  pS8 = L(-7.249056260830627156600112195061001036533E-1),
  pS9 =  L(1.055923570937755300061509030361395604448E-3),

  qS0 = L(-5.014859407482408326519083440151745519205E3),
  qS1 =  L(2.430653047950480068881028451580393430537E4),
  qS2 = L(-4.997904737193653607449250593976069726962E4),
  qS3 =  L(5.675712336110456923807959930107347511086E4),
  qS4 = L(-3.881523118339661268482937768522572588022E4),
  qS5 =  L(1.634202194895541569749717032234510811216E4),
  qS6 = L(-4.151452662440709301601820849901296953752E3),
  qS7 =  L(5.956050864057192019085175976175695342168E2),
  qS8 = L(-4.175375777334867025769346564600396877176E1),
  /* 1.000000000000000000000000000000000000000E0 */

  /* asin(0.5625 + x) = asin(0.5625) + x rS(x) / sS(x)
     -0.0625 <= x <= 0.0625
     peak relative error 3.3e-35  */
  rS0 = L(-5.619049346208901520945464704848780243887E0),
  rS1 =  L(4.460504162777731472539175700169871920352E1),
  rS2 = L(-1.317669505315409261479577040530751477488E2),
  rS3 =  L(1.626532582423661989632442410808596009227E2),
  rS4 = L(-3.144806644195158614904369445440583873264E1),
  rS5 = L(-9.806674443470740708765165604769099559553E1),
  rS6 =  L(5.708468492052010816555762842394927806920E1),
  rS7 =  L(1.396540499232262112248553357962639431922E1),
  rS8 = L(-1.126243289311910363001762058295832610344E1),
  rS9 = L(-4.956179821329901954211277873774472383512E-1),
  rS10 =  L(3.313227657082367169241333738391762525780E-1),

  sS0 = L(-4.645814742084009935700221277307007679325E0),
  sS1 =  L(3.879074822457694323970438316317961918430E1),
  sS2 = L(-1.221986588013474694623973554726201001066E2),
  sS3 =  L(1.658821150347718105012079876756201905822E2),
  sS4 = L(-4.804379630977558197953176474426239748977E1),
  sS5 = L(-1.004296417397316948114344573811562952793E2),
  sS6 =  L(7.530281592861320234941101403870010111138E1),
  sS7 =  L(1.270735595411673647119592092304357226607E1),
  sS8 = L(-1.815144839646376500705105967064792930282E1),
  sS9 = L(-7.821597334910963922204235247786840828217E-2),
  /*  1.000000000000000000000000000000000000000E0 */

 asinr5625 =  L(5.9740641664535021430381036628424864397707E-1);



_Float128
__ieee754_asinl (_Float128 x)
{
  _Float128 t, w, p, q, c, r, s;
  int32_t ix, sign, flag;
  ieee854_long_double_shape_type u;

  flag = 0;
  u.value = x;
  sign = u.parts32.w0;
  ix = sign & 0x7fffffff;
  u.parts32.w0 = ix;    /* |x| */
  if (ix >= 0x3fff0000)	/* |x|>= 1 */
    {
      if (ix == 0x3fff0000
	  && (u.parts32.w1 | u.parts32.w2 | u.parts32.w3) == 0)
	/* asin(1)=+-pi/2 with inexact */
	return x * pio2_hi + x * pio2_lo;
      return (x - x) / (x - x);	/* asin(|x|>1) is NaN */
    }
  else if (ix < 0x3ffe0000) /* |x| < 0.5 */
    {
      if (ix < 0x3fc60000) /* |x| < 2**-57 */
	{
	  math_check_force_underflow (x);
	  _Float128 force_inexact = huge + x;
	  math_force_eval (force_inexact);
	  return x;		/* return x with inexact if x!=0 */
	}
      else
	{
	  t = x * x;
	  /* Mark to use pS, qS later on.  */
	  flag = 1;
	}
    }
  else if (ix < 0x3ffe4000) /* 0.625 */
    {
      t = u.value - 0.5625;
      p = ((((((((((rS10 * t
		    + rS9) * t
		   + rS8) * t
		  + rS7) * t
		 + rS6) * t
		+ rS5) * t
	       + rS4) * t
	      + rS3) * t
	     + rS2) * t
	    + rS1) * t
	   + rS0) * t;

      q = ((((((((( t
		    + sS9) * t
		  + sS8) * t
		 + sS7) * t
		+ sS6) * t
	       + sS5) * t
	      + sS4) * t
	     + sS3) * t
	    + sS2) * t
	   + sS1) * t
	+ sS0;
      t = asinr5625 + p / q;
      if ((sign & 0x80000000) == 0)
	return t;
      else
	return -t;
    }
  else
    {
      /* 1 > |x| >= 0.625 */
      w = one - u.value;
      t = w * 0.5;
    }

  p = (((((((((pS9 * t
	       + pS8) * t
	      + pS7) * t
	     + pS6) * t
	    + pS5) * t
	   + pS4) * t
	  + pS3) * t
	 + pS2) * t
	+ pS1) * t
       + pS0) * t;

  q = (((((((( t
	      + qS8) * t
	     + qS7) * t
	    + qS6) * t
	   + qS5) * t
	  + qS4) * t
	 + qS3) * t
	+ qS2) * t
       + qS1) * t
    + qS0;

  if (flag) /* 2^-57 < |x| < 0.5 */
    {
      w = p / q;
      return x + x * w;
    }

  s = sqrtl (t);
  if (ix >= 0x3ffef333) /* |x| > 0.975 */
    {
      w = p / q;
      t = pio2_hi - (2.0 * (s + s * w) - pio2_lo);
    }
  else
    {
      u.value = s;
      u.parts32.w3 = 0;
      u.parts32.w2 = 0;
      w = u.value;
      c = (t - w * w) / (s + w);
      r = p / q;
      p = 2.0 * s * r - (pio2_lo - 2.0 * c);
      q = pio4_hi - 2.0 * w;
      t = pio4_hi - (p - q);
    }

  if ((sign & 0x80000000) == 0)
    return t;
  else
    return -t;
}
libm_alias_finite (__ieee754_asinl, __asinl)
