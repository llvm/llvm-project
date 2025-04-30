/* lgammal expanding around zeros.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <fenv_private.h>

static const long double lgamma_zeros[][2] =
  {
    { -0x2.74ff92c01f0d82acp+0L, 0x1.360cea0e5f8ed3ccp-68L },
    { -0x2.bf6821437b201978p+0L, -0x1.95a4b4641eaebf4cp-64L },
    { -0x3.24c1b793cb35efb8p+0L, -0xb.e699ad3d9ba6545p-68L },
    { -0x3.f48e2a8f85fca17p+0L, -0xd.4561291236cc321p-68L },
    { -0x4.0a139e16656030cp+0L, -0x3.9f0b0de18112ac18p-64L },
    { -0x4.fdd5de9bbabf351p+0L, -0xd.0aa4076988501d8p-68L },
    { -0x5.021a95fc2db64328p+0L, -0x2.4c56e595394decc8p-64L },
    { -0x5.ffa4bd647d0357ep+0L, 0x2.b129d342ce12071cp-64L },
    { -0x6.005ac9625f233b6p+0L, -0x7.c2d96d16385cb868p-68L },
    { -0x6.fff2fddae1bbff4p+0L, 0x2.9d949a3dc02de0cp-64L },
    { -0x7.000cff7b7f87adf8p+0L, 0x3.b7d23246787d54d8p-64L },
    { -0x7.fffe5fe05673c3c8p+0L, -0x2.9e82b522b0ca9d3p-64L },
    { -0x8.0001a01459fc9f6p+0L, -0xc.b3cec1cec857667p-68L },
    { -0x8.ffffd1c425e81p+0L, 0x3.79b16a8b6da6181cp-64L },
    { -0x9.00002e3bb47d86dp+0L, -0x6.d843fedc351deb78p-64L },
    { -0x9.fffffb606bdfdcdp+0L, -0x6.2ae77a50547c69dp-68L },
    { -0xa.0000049f93bb992p+0L, -0x7.b45d95e15441e03p-64L },
    { -0xa.ffffff9466e9f1bp+0L, -0x3.6dacd2adbd18d05cp-64L },
    { -0xb.0000006b9915316p+0L, 0x2.69a590015bf1b414p-64L },
    { -0xb.fffffff70893874p+0L, 0x7.821be533c2c36878p-64L },
    { -0xc.00000008f76c773p+0L, -0x1.567c0f0250f38792p-64L },
    { -0xc.ffffffff4f6dcf6p+0L, -0x1.7f97a5ffc757d548p-64L },
    { -0xd.00000000b09230ap+0L, 0x3.f997c22e46fc1c9p-64L },
    { -0xd.fffffffff36345bp+0L, 0x4.61e7b5c1f62ee89p-64L },
    { -0xe.000000000c9cba5p+0L, -0x4.5e94e75ec5718f78p-64L },
    { -0xe.ffffffffff28c06p+0L, -0xc.6604ef30371f89dp-68L },
    { -0xf.0000000000d73fap+0L, 0xc.6642f1bdf07a161p-68L },
    { -0xf.fffffffffff28cp+0L, -0x6.0c6621f512e72e5p-64L },
    { -0x1.000000000000d74p+4L, 0x6.0c6625ebdb406c48p-64L },
    { -0x1.0ffffffffffff356p+4L, -0x9.c47e7a93e1c46a1p-64L },
    { -0x1.1000000000000caap+4L, 0x9.c47e7a97778935ap-64L },
    { -0x1.1fffffffffffff4cp+4L, 0x1.3c31dcbecd2f74d4p-64L },
    { -0x1.20000000000000b4p+4L, -0x1.3c31dcbeca4c3b3p-64L },
    { -0x1.2ffffffffffffff6p+4L, -0x8.5b25cbf5f545ceep-64L },
    { -0x1.300000000000000ap+4L, 0x8.5b25cbf5f547e48p-64L },
    { -0x1.4p+4L, 0x7.950ae90080894298p-64L },
    { -0x1.4p+4L, -0x7.950ae9008089414p-64L },
    { -0x1.5p+4L, 0x5.c6e3bdb73d5c63p-68L },
    { -0x1.5p+4L, -0x5.c6e3bdb73d5c62f8p-68L },
    { -0x1.6p+4L, 0x4.338e5b6dfe14a518p-72L },
    { -0x1.6p+4L, -0x4.338e5b6dfe14a51p-72L },
    { -0x1.7p+4L, 0x2.ec368262c7033b3p-76L },
    { -0x1.7p+4L, -0x2.ec368262c7033b3p-76L },
    { -0x1.8p+4L, 0x1.f2cf01972f577ccap-80L },
    { -0x1.8p+4L, -0x1.f2cf01972f577ccap-80L },
    { -0x1.9p+4L, 0x1.3f3ccdd165fa8d4ep-84L },
    { -0x1.9p+4L, -0x1.3f3ccdd165fa8d4ep-84L },
    { -0x1.ap+4L, 0xc.4742fe35272cd1cp-92L },
    { -0x1.ap+4L, -0xc.4742fe35272cd1cp-92L },
    { -0x1.bp+4L, 0x7.46ac70b733a8c828p-96L },
    { -0x1.bp+4L, -0x7.46ac70b733a8c828p-96L },
    { -0x1.cp+4L, 0x4.2862898d42174ddp-100L },
    { -0x1.cp+4L, -0x4.2862898d42174ddp-100L },
    { -0x1.dp+4L, 0x2.4b3f31686b15af58p-104L },
    { -0x1.dp+4L, -0x2.4b3f31686b15af58p-104L },
    { -0x1.ep+4L, 0x1.3932c5047d60e60cp-108L },
    { -0x1.ep+4L, -0x1.3932c5047d60e60cp-108L },
    { -0x1.fp+4L, 0xa.1a6973c1fade217p-116L },
    { -0x1.fp+4L, -0xa.1a6973c1fade217p-116L },
    { -0x2p+4L, 0x5.0d34b9e0fd6f10b8p-120L },
    { -0x2p+4L, -0x5.0d34b9e0fd6f10b8p-120L },
    { -0x2.1p+4L, 0x2.73024a9ba1aa36a8p-124L },
  };

static const long double e_hi = 0x2.b7e151628aed2a6cp+0L;
static const long double e_lo = -0x1.408ea77f630b0c38p-64L;

/* Coefficients B_2k / 2k(2k-1) of x^-(2k-1) in Stirling's
   approximation to lgamma function.  */

static const long double lgamma_coeff[] =
  {
    0x1.5555555555555556p-4L,
    -0xb.60b60b60b60b60bp-12L,
    0x3.4034034034034034p-12L,
    -0x2.7027027027027028p-12L,
    0x3.72a3c5631fe46aep-12L,
    -0x7.daac36664f1f208p-12L,
    0x1.a41a41a41a41a41ap-8L,
    -0x7.90a1b2c3d4e5f708p-8L,
    0x2.dfd2c703c0cfff44p-4L,
    -0x1.6476701181f39edcp+0L,
    0xd.672219167002d3ap+0L,
    -0x9.cd9292e6660d55bp+4L,
    0x8.911a740da740da7p+8L,
    -0x8.d0cc570e255bf5ap+12L,
    0xa.8d1044d3708d1c2p+16L,
    -0xe.8844d8a169abbc4p+20L,
  };

#define NCOEFF (sizeof (lgamma_coeff) / sizeof (lgamma_coeff[0]))

/* Polynomial approximations to (|gamma(x)|-1)(x-n)/(x-x0), where n is
   the integer end-point of the half-integer interval containing x and
   x0 is the zero of lgamma in that half-integer interval.  Each
   polynomial is expressed in terms of x-xm, where xm is the midpoint
   of the interval for which the polynomial applies.  */

static const long double poly_coeff[] =
  {
    /* Interval [-2.125, -2] (polynomial degree 13).  */
    -0x1.0b71c5c54d42eb6cp+0L,
    -0xc.73a1dc05f349517p-4L,
    -0x1.ec841408528b6baep-4L,
    -0xe.37c9da26fc3b492p-4L,
    -0x1.03cd87c5178991ap-4L,
    -0xe.ae9ada65ece2f39p-4L,
    0x9.b1185505edac18dp-8L,
    -0xe.f28c130b54d3cb2p-4L,
    0x2.6ec1666cf44a63bp-4L,
    -0xf.57cb2774193bbd5p-4L,
    0x4.5ae64671a41b1c4p-4L,
    -0xf.f48ea8b5bd3a7cep-4L,
    0x6.7d73788a8d30ef58p-4L,
    -0x1.11e0e4b506bd272ep+0L,
    /* Interval [-2.25, -2.125] (polynomial degree 13).  */
    -0xf.2930890d7d675a8p-4L,
    -0xc.a5cfde054eab5cdp-4L,
    0x3.9c9e0fdebb0676e4p-4L,
    -0x1.02a5ad35605f0d8cp+0L,
    0x9.6e9b1185d0b92edp-4L,
    -0x1.4d8332f3d6a3959p+0L,
    0x1.1c0c8cacd0ced3eap+0L,
    -0x1.c9a6f592a67b1628p+0L,
    0x1.d7e9476f96aa4bd6p+0L,
    -0x2.921cedb488bb3318p+0L,
    0x2.e8b3fd6ca193e4c8p+0L,
    -0x3.cb69d9d6628e4a2p+0L,
    0x4.95f12c73b558638p+0L,
    -0x5.d392d0b97c02ab6p+0L,
    /* Interval [-2.375, -2.25] (polynomial degree 14).  */
    -0xd.7d28d505d618122p-4L,
    -0xe.69649a304098532p-4L,
    0xb.0d74a2827d055c5p-4L,
    -0x1.924b09228531c00ep+0L,
    0x1.d49b12bccee4f888p+0L,
    -0x3.0898bb7dbb21e458p+0L,
    0x4.207a6cad6fa10a2p+0L,
    -0x6.39ee630b46093ad8p+0L,
    0x8.e2e25211a3fb5ccp+0L,
    -0xd.0e85ccd8e79c08p+0L,
    0x1.2e45882bc17f9e16p+4L,
    -0x1.b8b6e841815ff314p+4L,
    0x2.7ff8bf7504fa04dcp+4L,
    -0x3.c192e9c903352974p+4L,
    0x5.8040b75f4ef07f98p+4L,
    /* Interval [-2.5, -2.375] (polynomial degree 15).  */
    -0xb.74ea1bcfff94b2cp-4L,
    -0x1.2a82bd590c375384p+0L,
    0x1.88020f828b968634p+0L,
    -0x3.32279f040eb80fa4p+0L,
    0x5.57ac825175943188p+0L,
    -0x9.c2aedcfe10f129ep+0L,
    0x1.12c132f2df02881ep+4L,
    -0x1.ea94e26c0b6ffa6p+4L,
    0x3.66b4a8bb0290013p+4L,
    -0x6.0cf735e01f5990bp+4L,
    0xa.c10a8db7ae99343p+4L,
    -0x1.31edb212b315feeap+8L,
    0x2.1f478592298b3ebp+8L,
    -0x3.c546da5957ace6ccp+8L,
    0x7.0e3d2a02579ba4bp+8L,
    -0xc.b1ea961c39302f8p+8L,
    /* Interval [-2.625, -2.5] (polynomial degree 16).  */
    -0x3.d10108c27ebafad4p-4L,
    0x1.cd557caff7d2b202p+0L,
    0x3.819b4856d3995034p+0L,
    0x6.8505cbad03dd3bd8p+0L,
    0xb.c1b2e653aa0b924p+0L,
    0x1.50a53a38f05f72d6p+4L,
    0x2.57ae00cbd06efb34p+4L,
    0x4.2b1563077a577e9p+4L,
    0x7.6989ed790138a7f8p+4L,
    0xd.2dd28417b4f8406p+4L,
    0x1.76e1b71f0710803ap+8L,
    0x2.9a7a096254ac032p+8L,
    0x4.a0e6109e2a039788p+8L,
    0x8.37ea17a93c877b2p+8L,
    0xe.9506a641143612bp+8L,
    0x1.b680ed4ea386d52p+12L,
    0x3.28a2130c8de0ae84p+12L,
    /* Interval [-2.75, -2.625] (polynomial degree 15).  */
    -0x6.b5d252a56e8a7548p-4L,
    0x1.28d60383da3ac72p+0L,
    0x1.db6513ada8a6703ap+0L,
    0x2.e217118f9d34aa7cp+0L,
    0x4.450112c5cbd6256p+0L,
    0x6.4af99151e972f92p+0L,
    0x9.2db598b5b183cd6p+0L,
    0xd.62bef9c9adcff6ap+0L,
    0x1.379f290d743d9774p+4L,
    0x1.c58271ff823caa26p+4L,
    0x2.93a871b87a06e73p+4L,
    0x3.bf9db66103d7ec98p+4L,
    0x5.73247c111fbf197p+4L,
    0x7.ec8b9973ba27d008p+4L,
    0xb.eca5f9619b39c03p+4L,
    0x1.18f2e46411c78b1cp+8L,
    /* Interval [-2.875, -2.75] (polynomial degree 14).  */
    -0x8.a41b1e4f36ff88ep-4L,
    0xc.da87d3b69dc0f34p-4L,
    0x1.1474ad5c36158ad2p+0L,
    0x1.761ecb90c5553996p+0L,
    0x1.d279bff9ae234f8p+0L,
    0x2.4e5d0055a16c5414p+0L,
    0x2.d57545a783902f8cp+0L,
    0x3.8514eec263aa9f98p+0L,
    0x4.5235e338245f6fe8p+0L,
    0x5.562b1ef200b256c8p+0L,
    0x6.8ec9782b93bd565p+0L,
    0x8.14baf4836483508p+0L,
    0x9.efaf35dc712ea79p+0L,
    0xc.8431f6a226507a9p+0L,
    0xf.80358289a768401p+0L,
    /* Interval [-3, -2.875] (polynomial degree 13).  */
    -0xa.046d667e468f3e4p-4L,
    0x9.70b88dcc006c216p-4L,
    0xa.a8a39421c86ce9p-4L,
    0xd.2f4d1363f321e89p-4L,
    0xd.ca9aa1a3ab2f438p-4L,
    0xf.cf09c31f05d02cbp-4L,
    0x1.04b133a195686a38p+0L,
    0x1.22b54799d0072024p+0L,
    0x1.2c5802b869a36ae8p+0L,
    0x1.4aadf23055d7105ep+0L,
    0x1.5794078dd45c55d6p+0L,
    0x1.7759069da18bcf0ap+0L,
    0x1.8e672cefa4623f34p+0L,
    0x1.b2acfa32c17145e6p+0L,
  };

static const size_t poly_deg[] =
  {
    13,
    13,
    14,
    15,
    16,
    15,
    14,
    13,
  };

static const size_t poly_end[] =
  {
    13,
    27,
    42,
    58,
    75,
    91,
    106,
    120,
  };

/* Compute sin (pi * X) for -0.25 <= X <= 0.5.  */

static long double
lg_sinpi (long double x)
{
  if (x <= 0.25L)
    return __sinl (M_PIl * x);
  else
    return __cosl (M_PIl * (0.5L - x));
}

/* Compute cos (pi * X) for -0.25 <= X <= 0.5.  */

static long double
lg_cospi (long double x)
{
  if (x <= 0.25L)
    return __cosl (M_PIl * x);
  else
    return __sinl (M_PIl * (0.5L - x));
}

/* Compute cot (pi * X) for -0.25 <= X <= 0.5.  */

static long double
lg_cotpi (long double x)
{
  return lg_cospi (x) / lg_sinpi (x);
}

/* Compute lgamma of a negative argument -33 < X < -2, setting
   *SIGNGAMP accordingly.  */

long double
__lgamma_negl (long double x, int *signgamp)
{
  /* Determine the half-integer region X lies in, handle exact
     integers and determine the sign of the result.  */
  int i = floorl (-2 * x);
  if ((i & 1) == 0 && i == -2 * x)
    return 1.0L / 0.0L;
  long double xn = ((i & 1) == 0 ? -i / 2 : (-i - 1) / 2);
  i -= 4;
  *signgamp = ((i & 2) == 0 ? -1 : 1);

  SET_RESTORE_ROUNDL (FE_TONEAREST);

  /* Expand around the zero X0 = X0_HI + X0_LO.  */
  long double x0_hi = lgamma_zeros[i][0], x0_lo = lgamma_zeros[i][1];
  long double xdiff = x - x0_hi - x0_lo;

  /* For arguments in the range -3 to -2, use polynomial
     approximations to an adjusted version of the gamma function.  */
  if (i < 2)
    {
      int j = floorl (-8 * x) - 16;
      long double xm = (-33 - 2 * j) * 0.0625L;
      long double x_adj = x - xm;
      size_t deg = poly_deg[j];
      size_t end = poly_end[j];
      long double g = poly_coeff[end];
      for (size_t j = 1; j <= deg; j++)
	g = g * x_adj + poly_coeff[end - j];
      return __log1pl (g * xdiff / (x - xn));
    }

  /* The result we want is log (sinpi (X0) / sinpi (X))
     + log (gamma (1 - X0) / gamma (1 - X)).  */
  long double x_idiff = fabsl (xn - x), x0_idiff = fabsl (xn - x0_hi - x0_lo);
  long double log_sinpi_ratio;
  if (x0_idiff < x_idiff * 0.5L)
    /* Use log not log1p to avoid inaccuracy from log1p of arguments
       close to -1.  */
    log_sinpi_ratio = __ieee754_logl (lg_sinpi (x0_idiff)
				      / lg_sinpi (x_idiff));
  else
    {
      /* Use log1p not log to avoid inaccuracy from log of arguments
	 close to 1.  X0DIFF2 has positive sign if X0 is further from
	 XN than X is from XN, negative sign otherwise.  */
      long double x0diff2 = ((i & 1) == 0 ? xdiff : -xdiff) * 0.5L;
      long double sx0d2 = lg_sinpi (x0diff2);
      long double cx0d2 = lg_cospi (x0diff2);
      log_sinpi_ratio = __log1pl (2 * sx0d2
				  * (-sx0d2 + cx0d2 * lg_cotpi (x_idiff)));
    }

  long double log_gamma_ratio;
  long double y0 = 1 - x0_hi;
  long double y0_eps = -x0_hi + (1 - y0) - x0_lo;
  long double y = 1 - x;
  long double y_eps = -x + (1 - y);
  /* We now wish to compute LOG_GAMMA_RATIO
     = log (gamma (Y0 + Y0_EPS) / gamma (Y + Y_EPS)).  XDIFF
     accurately approximates the difference Y0 + Y0_EPS - Y -
     Y_EPS.  Use Stirling's approximation.  First, we may need to
     adjust into the range where Stirling's approximation is
     sufficiently accurate.  */
  long double log_gamma_adj = 0;
  if (i < 8)
    {
      int n_up = (9 - i) / 2;
      long double ny0, ny0_eps, ny, ny_eps;
      ny0 = y0 + n_up;
      ny0_eps = y0 - (ny0 - n_up) + y0_eps;
      y0 = ny0;
      y0_eps = ny0_eps;
      ny = y + n_up;
      ny_eps = y - (ny - n_up) + y_eps;
      y = ny;
      y_eps = ny_eps;
      long double prodm1 = __lgamma_productl (xdiff, y - n_up, y_eps, n_up);
      log_gamma_adj = -__log1pl (prodm1);
    }
  long double log_gamma_high
    = (xdiff * __log1pl ((y0 - e_hi - e_lo + y0_eps) / e_hi)
       + (y - 0.5L + y_eps) * __log1pl (xdiff / y) + log_gamma_adj);
  /* Compute the sum of (B_2k / 2k(2k-1))(Y0^-(2k-1) - Y^-(2k-1)).  */
  long double y0r = 1 / y0, yr = 1 / y;
  long double y0r2 = y0r * y0r, yr2 = yr * yr;
  long double rdiff = -xdiff / (y * y0);
  long double bterm[NCOEFF];
  long double dlast = rdiff, elast = rdiff * yr * (yr + y0r);
  bterm[0] = dlast * lgamma_coeff[0];
  for (size_t j = 1; j < NCOEFF; j++)
    {
      long double dnext = dlast * y0r2 + elast;
      long double enext = elast * yr2;
      bterm[j] = dnext * lgamma_coeff[j];
      dlast = dnext;
      elast = enext;
    }
  long double log_gamma_low = 0;
  for (size_t j = 0; j < NCOEFF; j++)
    log_gamma_low += bterm[NCOEFF - 1 - j];
  log_gamma_ratio = log_gamma_high + log_gamma_low;

  return log_sinpi_ratio + log_gamma_ratio;
}
