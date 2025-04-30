/*							log10l.c
 *
 *	Common logarithm, 128-bit long double precision
 *
 *
 *
 * SYNOPSIS:
 *
 * long double x, y, log10l();
 *
 * y = log10l( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the base 10 logarithm of x.
 *
 * The argument is separated into its exponent and fractional
 * parts.  If the exponent is between -1 and +1, the logarithm
 * of the fraction is approximated by
 *
 *     log(1+x) = x - 0.5 x^2 + x^3 P(x)/Q(x).
 *
 * Otherwise, setting  z = 2(x-1)/x+1),
 *
 *     log(x) = z + z^3 P(z)/Q(z).
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0.5, 2.0     30000      2.3e-34     4.9e-35
 *    IEEE     exp(+-10000)  30000      1.0e-34     4.1e-35
 *
 * In the tests over the interval exp(+-10000), the logarithms
 * of the random arguments were uniformly distributed over
 * [-10000, +10000].
 *
 */

/*
   Cephes Math Library Release 2.2:  January, 1991
   Copyright 1984, 1991 by Stephen L. Moshier
   Adapted for glibc November, 2001

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, see <https://www.gnu.org/licenses/>.
 */

#include <math.h>
#include <math_private.h>
#include <libm-alias-finite.h>

/* Coefficients for ln(1+x) = x - x**2/2 + x**3 P(x)/Q(x)
 * 1/sqrt(2) <= x < sqrt(2)
 * Theoretical peak relative error = 5.3e-37,
 * relative peak error spread = 2.3e-14
 */
static const _Float128 P[13] =
{
  L(1.313572404063446165910279910527789794488E4),
  L(7.771154681358524243729929227226708890930E4),
  L(2.014652742082537582487669938141683759923E5),
  L(3.007007295140399532324943111654767187848E5),
  L(2.854829159639697837788887080758954924001E5),
  L(1.797628303815655343403735250238293741397E5),
  L(7.594356839258970405033155585486712125861E4),
  L(2.128857716871515081352991964243375186031E4),
  L(3.824952356185897735160588078446136783779E3),
  L(4.114517881637811823002128927449878962058E2),
  L(2.321125933898420063925789532045674660756E1),
  L(4.998469661968096229986658302195402690910E-1),
  L(1.538612243596254322971797716843006400388E-6)
};
static const _Float128 Q[12] =
{
  L(3.940717212190338497730839731583397586124E4),
  L(2.626900195321832660448791748036714883242E5),
  L(7.777690340007566932935753241556479363645E5),
  L(1.347518538384329112529391120390701166528E6),
  L(1.514882452993549494932585972882995548426E6),
  L(1.158019977462989115839826904108208787040E6),
  L(6.132189329546557743179177159925690841200E5),
  L(2.248234257620569139969141618556349415120E5),
  L(5.605842085972455027590989944010492125825E4),
  L(9.147150349299596453976674231612674085381E3),
  L(9.104928120962988414618126155557301584078E2),
  L(4.839208193348159620282142911143429644326E1)
/* 1.000000000000000000000000000000000000000E0L, */
};

/* Coefficients for log(x) = z + z^3 P(z^2)/Q(z^2),
 * where z = 2(x-1)/(x+1)
 * 1/sqrt(2) <= x < sqrt(2)
 * Theoretical peak relative error = 1.1e-35,
 * relative peak error spread 1.1e-9
 */
static const _Float128 R[6] =
{
  L(1.418134209872192732479751274970992665513E5),
 L(-8.977257995689735303686582344659576526998E4),
  L(2.048819892795278657810231591630928516206E4),
 L(-2.024301798136027039250415126250455056397E3),
  L(8.057002716646055371965756206836056074715E1),
 L(-8.828896441624934385266096344596648080902E-1)
};
static const _Float128 S[6] =
{
  L(1.701761051846631278975701529965589676574E6),
 L(-1.332535117259762928288745111081235577029E6),
  L(4.001557694070773974936904547424676279307E5),
 L(-5.748542087379434595104154610899551484314E4),
  L(3.998526750980007367835804959888064681098E3),
 L(-1.186359407982897997337150403816839480438E2)
/* 1.000000000000000000000000000000000000000E0L, */
};

static const _Float128
/* log10(2) */
L102A = L(0.3125),
L102B = L(-1.14700043360188047862611052755069732318101185E-2),
/* log10(e) */
L10EA = L(0.5),
L10EB = L(-6.570551809674817234887108108339491770560299E-2),
/* sqrt(2)/2 */
SQRTH = L(7.071067811865475244008443621048490392848359E-1);



/* Evaluate P[n] x^n  +  P[n-1] x^(n-1)  +  ...  +  P[0] */

static _Float128
neval (_Float128 x, const _Float128 *p, int n)
{
  _Float128 y;

  p += n;
  y = *p--;
  do
    {
      y = y * x + *p--;
    }
  while (--n > 0);
  return y;
}


/* Evaluate x^n+1  +  P[n] x^(n)  +  P[n-1] x^(n-1)  +  ...  +  P[0] */

static _Float128
deval (_Float128 x, const _Float128 *p, int n)
{
  _Float128 y;

  p += n;
  y = x + *p--;
  do
    {
      y = y * x + *p--;
    }
  while (--n > 0);
  return y;
}



_Float128
__ieee754_log10l (_Float128 x)
{
  _Float128 z;
  _Float128 y;
  int e;
  int64_t hx, lx;

/* Test for domain */
  GET_LDOUBLE_WORDS64 (hx, lx, x);
  if (((hx & 0x7fffffffffffffffLL) | lx) == 0)
    return (-1 / fabsl (x));		/* log10l(+-0)=-inf  */
  if (hx < 0)
    return (x - x) / (x - x);
  if (hx >= 0x7fff000000000000LL)
    return (x + x);

  if (x == 1)
    return 0;

/* separate mantissa from exponent */

/* Note, frexp is used so that denormal numbers
 * will be handled properly.
 */
  x = __frexpl (x, &e);


/* logarithm using log(x) = z + z**3 P(z)/Q(z),
 * where z = 2(x-1)/x+1)
 */
  if ((e > 2) || (e < -2))
    {
      if (x < SQRTH)
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
      y = x * (z * neval (z, R, 5) / deval (z, S, 5));
      goto done;
    }


/* logarithm using log(1+x) = x - .5x**2 + x**3 P(x)/Q(x) */

  if (x < SQRTH)
    {
      e -= 1;
      x = 2.0 * x - 1;	/*  2x - 1  */
    }
  else
    {
      x = x - 1;
    }
  z = x * x;
  y = x * (z * neval (x, P, 12) / deval (x, Q, 11));
  y = y - 0.5 * z;

done:

  /* Multiply log of fraction by log10(e)
   * and base 2 exponent by log10(2).
   */
  z = y * L10EB;
  z += x * L10EB;
  z += e * L102B;
  z += y * L10EA;
  z += x * L10EA;
  z += e * L102A;
  return (z);
}
libm_alias_finite (__ieee754_log10l, __log10l)
