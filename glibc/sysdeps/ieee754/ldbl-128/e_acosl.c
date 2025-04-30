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

/* __ieee754_acosl(x)
 * Method :
 *      acos(x)  = pi/2 - asin(x)
 *      acos(-x) = pi/2 + asin(x)
 * For |x| <= 0.375
 *      acos(x) = pi/2 - asin(x)
 * Between .375 and .5 the approximation is
 *      acos(0.4375 + x) = acos(0.4375) + x P(x) / Q(x)
 * Between .5 and .625 the approximation is
 *      acos(0.5625 + x) = acos(0.5625) + x rS(x) / sS(x)
 * For x > 0.625,
 *      acos(x) = 2 asin(sqrt((1-x)/2))
 *      computed with an extended precision square root in the leading term.
 * For x < -0.625
 *      acos(x) = pi - 2 asin(sqrt((1-|x|)/2))
 *
 * Special cases:
 *      if x is NaN, return x itself;
 *      if |x|>1, return NaN with invalid signal.
 *
 * Functions needed: sqrtl.
 */

#include <math.h>
#include <math_private.h>
#include <libm-alias-finite.h>

static const _Float128
  one = 1,
  pio2_hi = L(1.5707963267948966192313216916397514420986),
  pio2_lo = L(4.3359050650618905123985220130216759843812E-35),

  /* acos(0.5625 + x) = acos(0.5625) + x rS(x) / sS(x)
     -0.0625 <= x <= 0.0625
     peak relative error 3.3e-35  */

  rS0 =  L(5.619049346208901520945464704848780243887E0),
  rS1 = L(-4.460504162777731472539175700169871920352E1),
  rS2 =  L(1.317669505315409261479577040530751477488E2),
  rS3 = L(-1.626532582423661989632442410808596009227E2),
  rS4 =  L(3.144806644195158614904369445440583873264E1),
  rS5 =  L(9.806674443470740708765165604769099559553E1),
  rS6 = L(-5.708468492052010816555762842394927806920E1),
  rS7 = L(-1.396540499232262112248553357962639431922E1),
  rS8 =  L(1.126243289311910363001762058295832610344E1),
  rS9 =  L(4.956179821329901954211277873774472383512E-1),
  rS10 = L(-3.313227657082367169241333738391762525780E-1),

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
  /* 1.000000000000000000000000000000000000000E0 */

  acosr5625 = L(9.7338991014954640492751132535550279812151E-1),
  pimacosr5625 = L(2.1682027434402468335351320579240000860757E0),

  /* acos(0.4375 + x) = acos(0.4375) + x rS(x) / sS(x)
     -0.0625 <= x <= 0.0625
     peak relative error 2.1e-35  */

  P0 =  L(2.177690192235413635229046633751390484892E0),
  P1 = L(-2.848698225706605746657192566166142909573E1),
  P2 =  L(1.040076477655245590871244795403659880304E2),
  P3 = L(-1.400087608918906358323551402881238180553E2),
  P4 =  L(2.221047917671449176051896400503615543757E1),
  P5 =  L(9.643714856395587663736110523917499638702E1),
  P6 = L(-5.158406639829833829027457284942389079196E1),
  P7 = L(-1.578651828337585944715290382181219741813E1),
  P8 =  L(1.093632715903802870546857764647931045906E1),
  P9 =  L(5.448925479898460003048760932274085300103E-1),
  P10 = L(-3.315886001095605268470690485170092986337E-1),
  Q0 = L(-1.958219113487162405143608843774587557016E0),
  Q1 =  L(2.614577866876185080678907676023269360520E1),
  Q2 = L(-9.990858606464150981009763389881793660938E1),
  Q3 =  L(1.443958741356995763628660823395334281596E2),
  Q4 = L(-3.206441012484232867657763518369723873129E1),
  Q5 = L(-1.048560885341833443564920145642588991492E2),
  Q6 =  L(6.745883931909770880159915641984874746358E1),
  Q7 =  L(1.806809656342804436118449982647641392951E1),
  Q8 = L(-1.770150690652438294290020775359580915464E1),
  Q9 = L(-5.659156469628629327045433069052560211164E-1),
  /* 1.000000000000000000000000000000000000000E0 */

  acosr4375 = L(1.1179797320499710475919903296900511518755E0),
  pimacosr4375 = L(2.0236129215398221908706530535894517323217E0),

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
  qS8 = L(-4.175375777334867025769346564600396877176E1);
  /* 1.000000000000000000000000000000000000000E0 */

_Float128
__ieee754_acosl (_Float128 x)
{
  _Float128 z, r, w, p, q, s, t, f2;
  int32_t ix, sign;
  ieee854_long_double_shape_type u;

  u.value = x;
  sign = u.parts32.w0;
  ix = sign & 0x7fffffff;
  u.parts32.w0 = ix;		/* |x| */
  if (ix >= 0x3fff0000)		/* |x| >= 1 */
    {
      if (ix == 0x3fff0000
	  && (u.parts32.w1 | u.parts32.w2 | u.parts32.w3) == 0)
	{			/* |x| == 1 */
	  if ((sign & 0x80000000) == 0)
	    return 0.0;		/* acos(1) = 0  */
	  else
	    return (2.0 * pio2_hi) + (2.0 * pio2_lo);	/* acos(-1)= pi */
	}
      return (x - x) / (x - x);	/* acos(|x| > 1) is NaN */
    }
  else if (ix < 0x3ffe0000)	/* |x| < 0.5 */
    {
      if (ix < 0x3f8e0000)	/* |x| < 2**-113 */
	return pio2_hi + pio2_lo;
      if (ix < 0x3ffde000)	/* |x| < .4375 */
	{
	  /* Arcsine of x.  */
	  z = x * x;
	  p = (((((((((pS9 * z
		       + pS8) * z
		      + pS7) * z
		     + pS6) * z
		    + pS5) * z
		   + pS4) * z
		  + pS3) * z
		 + pS2) * z
		+ pS1) * z
	       + pS0) * z;
	  q = (((((((( z
		       + qS8) * z
		     + qS7) * z
		    + qS6) * z
		   + qS5) * z
		  + qS4) * z
		 + qS3) * z
		+ qS2) * z
	       + qS1) * z
	    + qS0;
	  r = x + x * p / q;
	  z = pio2_hi - (r - pio2_lo);
	  return z;
	}
      /* .4375 <= |x| < .5 */
      t = u.value - L(0.4375);
      p = ((((((((((P10 * t
		    + P9) * t
		   + P8) * t
		  + P7) * t
		 + P6) * t
		+ P5) * t
	       + P4) * t
	      + P3) * t
	     + P2) * t
	    + P1) * t
	   + P0) * t;

      q = (((((((((t
		   + Q9) * t
		  + Q8) * t
		 + Q7) * t
		+ Q6) * t
	       + Q5) * t
	      + Q4) * t
	     + Q3) * t
	    + Q2) * t
	   + Q1) * t
	+ Q0;
      r = p / q;
      if (sign & 0x80000000)
	r = pimacosr4375 - r;
      else
	r = acosr4375 + r;
      return r;
    }
  else if (ix < 0x3ffe4000)	/* |x| < 0.625 */
    {
      t = u.value - L(0.5625);
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

      q = (((((((((t
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
      if (sign & 0x80000000)
	r = pimacosr5625 - p / q;
      else
	r = acosr5625 + p / q;
      return r;
    }
  else
    {				/* |x| >= .625 */
      z = (one - u.value) * 0.5;
      s = sqrtl (z);
      /* Compute an extended precision square root from
	 the Newton iteration  s -> 0.5 * (s + z / s).
	 The change w from s to the improved value is
	    w = 0.5 * (s + z / s) - s  = (s^2 + z)/2s - s = (z - s^2)/2s.
	  Express s = f1 + f2 where f1 * f1 is exactly representable.
	  w = (z - s^2)/2s = (z - f1^2 - 2 f1 f2 - f2^2)/2s .
	  s + w has extended precision.  */
      u.value = s;
      u.parts32.w2 = 0;
      u.parts32.w3 = 0;
      f2 = s - u.value;
      w = z - u.value * u.value;
      w = w - 2.0 * u.value * f2;
      w = w - f2 * f2;
      w = w / (2.0 * s);
      /* Arcsine of s.  */
      p = (((((((((pS9 * z
		   + pS8) * z
		  + pS7) * z
		 + pS6) * z
		+ pS5) * z
	       + pS4) * z
	      + pS3) * z
	     + pS2) * z
	    + pS1) * z
	   + pS0) * z;
      q = (((((((( z
		   + qS8) * z
		 + qS7) * z
		+ qS6) * z
	       + qS5) * z
	      + qS4) * z
	     + qS3) * z
	    + qS2) * z
	   + qS1) * z
	+ qS0;
      r = s + (w + s * p / q);

      if (sign & 0x80000000)
	w = pio2_hi + (pio2_lo - r);
      else
	w = r;
      return 2.0 * w;
    }
}
libm_alias_finite (__ieee754_acosl, __acosl)
