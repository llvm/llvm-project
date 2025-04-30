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
#include <float.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <libm-alias-ldouble.h>

/* exp(x) - 1 = x + 0.5 x^2 + x^3 P(x)/Q(x)
   -.5 ln 2  <  x  <  .5 ln 2
   Theoretical peak relative error = 8.1e-36  */

static const _Float128
  P0 = L(2.943520915569954073888921213330863757240E8),
  P1 = L(-5.722847283900608941516165725053359168840E7),
  P2 = L(8.944630806357575461578107295909719817253E6),
  P3 = L(-7.212432713558031519943281748462837065308E5),
  P4 = L(4.578962475841642634225390068461943438441E4),
  P5 = L(-1.716772506388927649032068540558788106762E3),
  P6 = L(4.401308817383362136048032038528753151144E1),
  P7 = L(-4.888737542888633647784737721812546636240E-1),
  Q0 = L(1.766112549341972444333352727998584753865E9),
  Q1 = L(-7.848989743695296475743081255027098295771E8),
  Q2 = L(1.615869009634292424463780387327037251069E8),
  Q3 = L(-2.019684072836541751428967854947019415698E7),
  Q4 = L(1.682912729190313538934190635536631941751E6),
  Q5 = L(-9.615511549171441430850103489315371768998E4),
  Q6 = L(3.697714952261803935521187272204485251835E3),
  Q7 = L(-8.802340681794263968892934703309274564037E1),
  /* Q8 = 1.000000000000000000000000000000000000000E0 */
/* C1 + C2 = ln 2 */

  C1 = L(6.93145751953125E-1),
  C2 = L(1.428606820309417232121458176568075500134E-6),
/* ln 2^-114 */
  minarg = L(-7.9018778583833765273564461846232128760607E1), big = L(1e4932);


_Float128
__expm1l (_Float128 x)
{
  _Float128 px, qx, xx;
  int32_t ix, sign;
  ieee854_long_double_shape_type u;
  int k;

  /* Detect infinity and NaN.  */
  u.value = x;
  ix = u.parts32.w0;
  sign = ix & 0x80000000;
  ix &= 0x7fffffff;
  if (!sign && ix >= 0x40060000)
    {
      /* If num is positive and exp >= 6 use plain exp.  */
      return __expl (x);
    }
  if (ix >= 0x7fff0000)
    {
      /* Infinity (which must be negative infinity). */
      if (((ix & 0xffff) | u.parts32.w1 | u.parts32.w2 | u.parts32.w3) == 0)
	return -1;
      /* NaN.  Invalid exception if signaling.  */
      return x + x;
    }

  /* expm1(+- 0) = +- 0.  */
  if ((ix == 0) && (u.parts32.w1 | u.parts32.w2 | u.parts32.w3) == 0)
    return x;

  /* Minimum value.  */
  if (x < minarg)
    return (4.0/big - 1);

  /* Avoid internal underflow when result does not underflow, while
     ensuring underflow (without returning a zero of the wrong sign)
     when the result does underflow.  */
  if (fabsl (x) < L(0x1p-113))
    {
      math_check_force_underflow (x);
      return x;
    }

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

  px = __ldexpl (1, k);
  x = px * qx + (px - 1.0);
  return x;
}
libm_hidden_def (__expm1l)
libm_alias_ldouble (__expm1, expm1)
