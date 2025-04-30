/*							log1pl.c
 *
 *      Relative error logarithm
 *	Natural logarithm of 1+x, 128-bit long double precision
 *
 *
 *
 * SYNOPSIS:
 *
 * long double x, y, log1pl();
 *
 * y = log1pl( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the base e (2.718...) logarithm of 1+x.
 *
 * The argument 1+x is separated into its exponent and fractional
 * parts.  If the exponent is between -1 and +1, the logarithm
 * of the fraction is approximated by
 *
 *     log(1+x) = x - 0.5 x^2 + x^3 P(x)/Q(x).
 *
 * Otherwise, setting  z = 2(w-1)/(w+1),
 *
 *     log(w) = z + z^3 P(z)/Q(z).
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      -1, 8       100000      1.9e-34     4.3e-35
 */

/* Copyright 2001 by Stephen L. Moshier

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


#include <float.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>

/* Coefficients for log(1+x) = x - x^2 / 2 + x^3 P(x)/Q(x)
 * 1/sqrt(2) <= 1+x < sqrt(2)
 * Theoretical peak relative error = 5.3e-37,
 * relative peak error spread = 2.3e-14
 */
static const _Float128
  P12 = L(1.538612243596254322971797716843006400388E-6),
  P11 = L(4.998469661968096229986658302195402690910E-1),
  P10 = L(2.321125933898420063925789532045674660756E1),
  P9 = L(4.114517881637811823002128927449878962058E2),
  P8 = L(3.824952356185897735160588078446136783779E3),
  P7 = L(2.128857716871515081352991964243375186031E4),
  P6 = L(7.594356839258970405033155585486712125861E4),
  P5 = L(1.797628303815655343403735250238293741397E5),
  P4 = L(2.854829159639697837788887080758954924001E5),
  P3 = L(3.007007295140399532324943111654767187848E5),
  P2 = L(2.014652742082537582487669938141683759923E5),
  P1 = L(7.771154681358524243729929227226708890930E4),
  P0 = L(1.313572404063446165910279910527789794488E4),
  /* Q12 = 1.000000000000000000000000000000000000000E0L, */
  Q11 = L(4.839208193348159620282142911143429644326E1),
  Q10 = L(9.104928120962988414618126155557301584078E2),
  Q9 = L(9.147150349299596453976674231612674085381E3),
  Q8 = L(5.605842085972455027590989944010492125825E4),
  Q7 = L(2.248234257620569139969141618556349415120E5),
  Q6 = L(6.132189329546557743179177159925690841200E5),
  Q5 = L(1.158019977462989115839826904108208787040E6),
  Q4 = L(1.514882452993549494932585972882995548426E6),
  Q3 = L(1.347518538384329112529391120390701166528E6),
  Q2 = L(7.777690340007566932935753241556479363645E5),
  Q1 = L(2.626900195321832660448791748036714883242E5),
  Q0 = L(3.940717212190338497730839731583397586124E4);

/* Coefficients for log(x) = z + z^3 P(z^2)/Q(z^2),
 * where z = 2(x-1)/(x+1)
 * 1/sqrt(2) <= x < sqrt(2)
 * Theoretical peak relative error = 1.1e-35,
 * relative peak error spread 1.1e-9
 */
static const _Float128
  R5 = L(-8.828896441624934385266096344596648080902E-1),
  R4 = L(8.057002716646055371965756206836056074715E1),
  R3 = L(-2.024301798136027039250415126250455056397E3),
  R2 = L(2.048819892795278657810231591630928516206E4),
  R1 = L(-8.977257995689735303686582344659576526998E4),
  R0 = L(1.418134209872192732479751274970992665513E5),
  /* S6 = 1.000000000000000000000000000000000000000E0L, */
  S5 = L(-1.186359407982897997337150403816839480438E2),
  S4 = L(3.998526750980007367835804959888064681098E3),
  S3 = L(-5.748542087379434595104154610899551484314E4),
  S2 = L(4.001557694070773974936904547424676279307E5),
  S1 = L(-1.332535117259762928288745111081235577029E6),
  S0 = L(1.701761051846631278975701529965589676574E6);

/* C1 + C2 = ln 2 */
static const _Float128 C1 = L(6.93145751953125E-1);
static const _Float128 C2 = L(1.428606820309417232121458176568075500134E-6);

static const _Float128 sqrth = L(0.7071067811865475244008443621048490392848);
/* ln (2^16384 * (1 - 2^-113)) */
static const _Float128 zero = 0;

_Float128
__log1pl (_Float128 xm1)
{
  _Float128 x, y, z, r, s;
  ieee854_long_double_shape_type u;
  int32_t hx;
  int e;

  /* Test for NaN or infinity input. */
  u.value = xm1;
  hx = u.parts32.w0;
  if ((hx & 0x7fffffff) >= 0x7fff0000)
    return xm1 + fabsl (xm1);

  /* log1p(+- 0) = +- 0.  */
  if (((hx & 0x7fffffff) == 0)
      && (u.parts32.w1 | u.parts32.w2 | u.parts32.w3) == 0)
    return xm1;

  if ((hx & 0x7fffffff) < 0x3f8e0000)
    {
      math_check_force_underflow (xm1);
      if ((int) xm1 == 0)
	return xm1;
    }

  if (xm1 >= L(0x1p113))
    x = xm1;
  else
    x = xm1 + 1;

  /* log1p(-1) = -inf */
  if (x <= 0)
    {
      if (x == 0)
	return (-1 / zero);  /* log1p(-1) = -inf */
      else
	return (zero / (x - x));
    }

  /* Separate mantissa from exponent.  */

  /* Use frexp used so that denormal numbers will be handled properly.  */
  x = __frexpl (x, &e);

  /* Logarithm using log(x) = z + z^3 P(z^2)/Q(z^2),
     where z = 2(x-1)/x+1).  */
  if ((e > 2) || (e < -2))
    {
      if (x < sqrth)
	{			/* 2( 2x-1 )/( 2x+1 ) */
	  e -= 1;
	  z = x - L(0.5);
	  y = L(0.5) * z + L(0.5);
	}
      else
	{			/*  2 (x-1)/(x+1)   */
	  z = x - L(0.5);
	  z -= L(0.5);
	  y = L(0.5) * x + L(0.5);
	}
      x = z / y;
      z = x * x;
      r = ((((R5 * z
	      + R4) * z
	     + R3) * z
	    + R2) * z
	   + R1) * z
	+ R0;
      s = (((((z
	       + S5) * z
	      + S4) * z
	     + S3) * z
	    + S2) * z
	   + S1) * z
	+ S0;
      z = x * (z * r / s);
      z = z + e * C2;
      z = z + x;
      z = z + e * C1;
      return (z);
    }


  /* Logarithm using log(1+x) = x - .5x^2 + x^3 P(x)/Q(x). */

  if (x < sqrth)
    {
      e -= 1;
      if (e != 0)
	x = 2 * x - 1;	/*  2x - 1  */
      else
	x = xm1;
    }
  else
    {
      if (e != 0)
	x = x - 1;
      else
	x = xm1;
    }
  z = x * x;
  r = (((((((((((P12 * x
		 + P11) * x
		+ P10) * x
	       + P9) * x
	      + P8) * x
	     + P7) * x
	    + P6) * x
	   + P5) * x
	  + P4) * x
	 + P3) * x
	+ P2) * x
       + P1) * x
    + P0;
  s = (((((((((((x
		 + Q11) * x
		+ Q10) * x
	       + Q9) * x
	      + Q8) * x
	     + Q7) * x
	    + Q6) * x
	   + Q5) * x
	  + Q4) * x
	 + Q3) * x
	+ Q2) * x
       + Q1) * x
    + Q0;
  y = x * (z * r / s);
  y = y + e * C2;
  z = y - L(0.5) * z;
  z = z + x;
  z = z + e * C1;
  return (z);
}
