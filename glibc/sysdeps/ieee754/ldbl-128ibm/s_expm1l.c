/*							expm1l.c
 *
 *	Exponential function, minus 1
 *      128-bit long double precision
 *
 *
 *
 * SYNOPSIS:
 *
 * long double x, y, expm1l();
 *
 * y = expm1l( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns e (2.71828...) raised to the x power, minus one.
 *
 * Range reduction is accomplished by separating the argument
 * into an integer k and fraction f such that
 *
 *     x    k  f
 *    e  = 2  e.
 *
 * An expansion x + .5 x^2 + x^3 R(x) approximates exp(f) - 1
 * in the basic range [-0.5 ln 2, 0.5 ln 2].
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE    -79,+MAXLOG    100,000     1.7e-34     4.5e-35
 *
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

#include <errno.h>
#include <math.h>
#include <math_private.h>
#include <math_ldbl_opt.h>

/* exp(x) - 1 = x + 0.5 x^2 + x^3 P(x)/Q(x)
   -.5 ln 2  <  x  <  .5 ln 2
   Theoretical peak relative error = 8.1e-36  */

static const long double
  P0 = 2.943520915569954073888921213330863757240E8L,
  P1 = -5.722847283900608941516165725053359168840E7L,
  P2 = 8.944630806357575461578107295909719817253E6L,
  P3 = -7.212432713558031519943281748462837065308E5L,
  P4 = 4.578962475841642634225390068461943438441E4L,
  P5 = -1.716772506388927649032068540558788106762E3L,
  P6 = 4.401308817383362136048032038528753151144E1L,
  P7 = -4.888737542888633647784737721812546636240E-1L,
  Q0 = 1.766112549341972444333352727998584753865E9L,
  Q1 = -7.848989743695296475743081255027098295771E8L,
  Q2 = 1.615869009634292424463780387327037251069E8L,
  Q3 = -2.019684072836541751428967854947019415698E7L,
  Q4 = 1.682912729190313538934190635536631941751E6L,
  Q5 = -9.615511549171441430850103489315371768998E4L,
  Q6 = 3.697714952261803935521187272204485251835E3L,
  Q7 = -8.802340681794263968892934703309274564037E1L,
  /* Q8 = 1.000000000000000000000000000000000000000E0 */
/* C1 + C2 = ln 2 */

  C1 = 6.93145751953125E-1L,
  C2 = 1.428606820309417232121458176568075500134E-6L,
/* ln 2^-114 */
  minarg = -7.9018778583833765273564461846232128760607E1L, big = 1e290L;


long double
__expm1l (long double x)
{
  long double px, qx, xx;
  int32_t ix, lx, sign;
  int k;
  double xhi;

  /* Detect infinity and NaN.  */
  xhi = ldbl_high (x);
  EXTRACT_WORDS (ix, lx, xhi);
  sign = ix & 0x80000000;
  ix &= 0x7fffffff;
  if (!sign && ix >= 0x40600000)
    return __expl (x);
  if (ix >= 0x7ff00000)
    {
      /* Infinity (which must be negative infinity). */
      if (((ix - 0x7ff00000) | lx) == 0)
	return -1.0L;
      /* NaN.  Invalid exception if signaling.  */
      return x + x;
    }

  /* expm1(+- 0) = +- 0.  */
  if ((ix | lx) == 0)
    return x;

  /* Minimum value.  */
  if (x < minarg)
    return (4.0/big - 1.0L);

  /* Express x = ln 2 (k + remainder), remainder not exceeding 1/2. */
  xx = C1 + C2;			/* ln 2. */
  px = floorl (0.5 + x / xx);
  k = px;
  /* remainder times ln 2 */
  x -= px * C1;
  x -= px * C2;

  /* Approximate exp(remainder ln 2).  */
  px = (((((((P7 * x
	      + P6) * x
	     + P5) * x + P4) * x + P3) * x + P2) * x + P1) * x + P0) * x;

  qx = (((((((x
	      + Q7) * x
	     + Q6) * x + Q5) * x + Q4) * x + Q3) * x + Q2) * x + Q1) * x + Q0;

  xx = x * x;
  qx = x + (0.5 * xx + xx * px / qx);

  /* exp(x) = exp(k ln 2) exp(remainder ln 2) = 2^k exp(remainder ln 2).

  We have qx = exp(remainder ln 2) - 1, so
  exp(x) - 1 = 2^k (qx + 1) - 1
             = 2^k qx + 2^k - 1.  */

  px = __ldexpl (1.0L, k);
  x = px * qx + (px - 1.0);
  return x;
}
libm_hidden_def (__expm1l)
long_double_symbol (libm, __expm1l, expm1l);
