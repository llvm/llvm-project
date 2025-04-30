/* lgamma expanding around zeros.
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
#include <math-narrow-eval.h>
#include <math_private.h>
#include <fenv_private.h>

static const double lgamma_zeros[][2] =
  {
    { -0x2.74ff92c01f0d8p+0, -0x2.abec9f315f1ap-56 },
    { -0x2.bf6821437b202p+0, 0x6.866a5b4b9be14p-56 },
    { -0x3.24c1b793cb35ep+0, -0xf.b8be699ad3d98p-56 },
    { -0x3.f48e2a8f85fcap+0, -0x1.70d4561291237p-56 },
    { -0x4.0a139e1665604p+0, 0xf.3c60f4f21e7fp-56 },
    { -0x4.fdd5de9bbabf4p+0, 0xa.ef2f55bf89678p-56 },
    { -0x5.021a95fc2db64p+0, -0x3.2a4c56e595394p-56 },
    { -0x5.ffa4bd647d034p+0, -0x1.7dd4ed62cbd32p-52 },
    { -0x6.005ac9625f234p+0, 0x4.9f83d2692e9c8p-56 },
    { -0x6.fff2fddae1bcp+0, 0xc.29d949a3dc03p-60 },
    { -0x7.000cff7b7f87cp+0, 0x1.20bb7d2324678p-52 },
    { -0x7.fffe5fe05673cp+0, -0x3.ca9e82b522b0cp-56 },
    { -0x8.0001a01459fc8p+0, -0x1.f60cb3cec1cedp-52 },
    { -0x8.ffffd1c425e8p+0, -0xf.fc864e9574928p-56 },
    { -0x9.00002e3bb47d8p+0, -0x6.d6d843fedc35p-56 },
    { -0x9.fffffb606bep+0, 0x2.32f9d51885afap-52 },
    { -0xa.0000049f93bb8p+0, -0x1.927b45d95e154p-52 },
    { -0xa.ffffff9466eap+0, 0xe.4c92532d5243p-56 },
    { -0xb.0000006b9915p+0, -0x3.15d965a6ffea4p-52 },
    { -0xb.fffffff708938p+0, -0x7.387de41acc3d4p-56 },
    { -0xc.00000008f76c8p+0, 0x8.cea983f0fdafp-56 },
    { -0xc.ffffffff4f6ep+0, 0x3.09e80685a0038p-52 },
    { -0xd.00000000b092p+0, -0x3.09c06683dd1bap-52 },
    { -0xd.fffffffff3638p+0, 0x3.a5461e7b5c1f6p-52 },
    { -0xe.000000000c9c8p+0, -0x3.a545e94e75ec6p-52 },
    { -0xe.ffffffffff29p+0, 0x3.f9f399fb10cfcp-52 },
    { -0xf.0000000000d7p+0, -0x3.f9f399bd0e42p-52 },
    { -0xf.fffffffffff28p+0, -0xc.060c6621f513p-56 },
    { -0x1.000000000000dp+4, -0x7.3f9f399da1424p-52 },
    { -0x1.0ffffffffffffp+4, -0x3.569c47e7a93e2p-52 },
    { -0x1.1000000000001p+4, 0x3.569c47e7a9778p-52 },
    { -0x1.2p+4, 0xb.413c31dcbecdp-56 },
    { -0x1.2p+4, -0xb.413c31dcbeca8p-56 },
    { -0x1.3p+4, 0x9.7a4da340a0ab8p-60 },
    { -0x1.3p+4, -0x9.7a4da340a0ab8p-60 },
    { -0x1.4p+4, 0x7.950ae90080894p-64 },
    { -0x1.4p+4, -0x7.950ae90080894p-64 },
    { -0x1.5p+4, 0x5.c6e3bdb73d5c8p-68 },
    { -0x1.5p+4, -0x5.c6e3bdb73d5c8p-68 },
    { -0x1.6p+4, 0x4.338e5b6dfe14cp-72 },
    { -0x1.6p+4, -0x4.338e5b6dfe14cp-72 },
    { -0x1.7p+4, 0x2.ec368262c7034p-76 },
    { -0x1.7p+4, -0x2.ec368262c7034p-76 },
    { -0x1.8p+4, 0x1.f2cf01972f578p-80 },
    { -0x1.8p+4, -0x1.f2cf01972f578p-80 },
    { -0x1.9p+4, 0x1.3f3ccdd165fa9p-84 },
    { -0x1.9p+4, -0x1.3f3ccdd165fa9p-84 },
    { -0x1.ap+4, 0xc.4742fe35272dp-92 },
    { -0x1.ap+4, -0xc.4742fe35272dp-92 },
    { -0x1.bp+4, 0x7.46ac70b733a8cp-96 },
    { -0x1.bp+4, -0x7.46ac70b733a8cp-96 },
    { -0x1.cp+4, 0x4.2862898d42174p-100 },
  };

static const double e_hi = 0x2.b7e151628aed2p+0, e_lo = 0xa.6abf7158809dp-56;

/* Coefficients B_2k / 2k(2k-1) of x^-(2k-1) in Stirling's
   approximation to lgamma function.  */

static const double lgamma_coeff[] =
  {
    0x1.5555555555555p-4,
    -0xb.60b60b60b60b8p-12,
    0x3.4034034034034p-12,
    -0x2.7027027027028p-12,
    0x3.72a3c5631fe46p-12,
    -0x7.daac36664f1f4p-12,
    0x1.a41a41a41a41ap-8,
    -0x7.90a1b2c3d4e6p-8,
    0x2.dfd2c703c0dp-4,
    -0x1.6476701181f3ap+0,
    0xd.672219167003p+0,
    -0x9.cd9292e6660d8p+4,
  };

#define NCOEFF (sizeof (lgamma_coeff) / sizeof (lgamma_coeff[0]))

/* Polynomial approximations to (|gamma(x)|-1)(x-n)/(x-x0), where n is
   the integer end-point of the half-integer interval containing x and
   x0 is the zero of lgamma in that half-integer interval.  Each
   polynomial is expressed in terms of x-xm, where xm is the midpoint
   of the interval for which the polynomial applies.  */

static const double poly_coeff[] =
  {
    /* Interval [-2.125, -2] (polynomial degree 10).  */
    -0x1.0b71c5c54d42fp+0,
    -0xc.73a1dc05f3758p-4,
    -0x1.ec84140851911p-4,
    -0xe.37c9da23847e8p-4,
    -0x1.03cd87cdc0ac6p-4,
    -0xe.ae9aedce12eep-4,
    0x9.b11a1780cfd48p-8,
    -0xe.f25fc460bdebp-4,
    0x2.6e984c61ca912p-4,
    -0xf.83fea1c6d35p-4,
    0x4.760c8c8909758p-4,
    /* Interval [-2.25, -2.125] (polynomial degree 11).  */
    -0xf.2930890d7d678p-4,
    -0xc.a5cfde054eaa8p-4,
    0x3.9c9e0fdebd99cp-4,
    -0x1.02a5ad35619d9p+0,
    0x9.6e9b1167c164p-4,
    -0x1.4d8332eba090ap+0,
    0x1.1c0c94b1b2b6p+0,
    -0x1.c9a70d138c74ep+0,
    0x1.d7d9cf1d4c196p+0,
    -0x2.91fbf4cd6abacp+0,
    0x2.f6751f74b8ff8p+0,
    -0x3.e1bb7b09e3e76p+0,
    /* Interval [-2.375, -2.25] (polynomial degree 12).  */
    -0xd.7d28d505d618p-4,
    -0xe.69649a3040958p-4,
    0xb.0d74a2827cd6p-4,
    -0x1.924b09228a86ep+0,
    0x1.d49b12bcf6175p+0,
    -0x3.0898bb530d314p+0,
    0x4.207a6be8fda4cp+0,
    -0x6.39eef56d4e9p+0,
    0x8.e2e42acbccec8p+0,
    -0xd.0d91c1e596a68p+0,
    0x1.2e20d7099c585p+4,
    -0x1.c4eb6691b4ca9p+4,
    0x2.96a1a11fd85fep+4,
    /* Interval [-2.5, -2.375] (polynomial degree 13).  */
    -0xb.74ea1bcfff948p-4,
    -0x1.2a82bd590c376p+0,
    0x1.88020f828b81p+0,
    -0x3.32279f040d7aep+0,
    0x5.57ac8252ce868p+0,
    -0x9.c2aedd093125p+0,
    0x1.12c132716e94cp+4,
    -0x1.ea94dfa5c0a6dp+4,
    0x3.66b61abfe858cp+4,
    -0x6.0cfceb62a26e4p+4,
    0xa.beeba09403bd8p+4,
    -0x1.3188d9b1b288cp+8,
    0x2.37f774dd14c44p+8,
    -0x3.fdf0a64cd7136p+8,
    /* Interval [-2.625, -2.5] (polynomial degree 13).  */
    -0x3.d10108c27ebbp-4,
    0x1.cd557caff7d2fp+0,
    0x3.819b4856d36cep+0,
    0x6.8505cbacfc42p+0,
    0xb.c1b2e6567a4dp+0,
    0x1.50a53a3ce6c73p+4,
    0x2.57adffbb1ec0cp+4,
    0x4.2b15549cf400cp+4,
    0x7.698cfd82b3e18p+4,
    0xd.2decde217755p+4,
    0x1.7699a624d07b9p+8,
    0x2.98ecf617abbfcp+8,
    0x4.d5244d44d60b4p+8,
    0x8.e962bf7395988p+8,
    /* Interval [-2.75, -2.625] (polynomial degree 12).  */
    -0x6.b5d252a56e8a8p-4,
    0x1.28d60383da3a6p+0,
    0x1.db6513ada89bep+0,
    0x2.e217118fa8c02p+0,
    0x4.450112c651348p+0,
    0x6.4af990f589b8cp+0,
    0x9.2db5963d7a238p+0,
    0xd.62c03647da19p+0,
    0x1.379f81f6416afp+4,
    0x1.c5618b4fdb96p+4,
    0x2.9342d0af2ac4ep+4,
    0x3.d9cdf56d2b186p+4,
    0x5.ab9f91d5a27a4p+4,
    /* Interval [-2.875, -2.75] (polynomial degree 11).  */
    -0x8.a41b1e4f36ff8p-4,
    0xc.da87d3b69dbe8p-4,
    0x1.1474ad5c36709p+0,
    0x1.761ecb90c8c5cp+0,
    0x1.d279bff588826p+0,
    0x2.4e5d003fb36a8p+0,
    0x2.d575575566842p+0,
    0x3.85152b0d17756p+0,
    0x4.5213d921ca13p+0,
    0x5.55da7dfcf69c4p+0,
    0x6.acef729b9404p+0,
    0x8.483cc21dd0668p+0,
    /* Interval [-3, -2.875] (polynomial degree 11).  */
    -0xa.046d667e468f8p-4,
    0x9.70b88dcc006cp-4,
    0xa.a8a39421c94dp-4,
    0xd.2f4d1363f98ep-4,
    0xd.ca9aa19975b7p-4,
    0xf.cf09c2f54404p-4,
    0x1.04b1365a9adfcp+0,
    0x1.22b54ef213798p+0,
    0x1.2c52c25206bf5p+0,
    0x1.4aa3d798aace4p+0,
    0x1.5c3f278b504e3p+0,
    0x1.7e08292cc347bp+0,
  };

static const size_t poly_deg[] =
  {
    10,
    11,
    12,
    13,
    13,
    12,
    11,
    11,
  };

static const size_t poly_end[] =
  {
    10,
    22,
    35,
    49,
    63,
    76,
    88,
    100,
  };

/* Compute sin (pi * X) for -0.25 <= X <= 0.5.  */

static double
lg_sinpi (double x)
{
  if (x <= 0.25)
    return __sin (M_PI * x);
  else
    return __cos (M_PI * (0.5 - x));
}

/* Compute cos (pi * X) for -0.25 <= X <= 0.5.  */

static double
lg_cospi (double x)
{
  if (x <= 0.25)
    return __cos (M_PI * x);
  else
    return __sin (M_PI * (0.5 - x));
}

/* Compute cot (pi * X) for -0.25 <= X <= 0.5.  */

static double
lg_cotpi (double x)
{
  return lg_cospi (x) / lg_sinpi (x);
}

/* Compute lgamma of a negative argument -28 < X < -2, setting
   *SIGNGAMP accordingly.  */

double
__lgamma_neg (double x, int *signgamp)
{
  /* Determine the half-integer region X lies in, handle exact
     integers and determine the sign of the result.  */
  int i = floor (-2 * x);
  if ((i & 1) == 0 && i == -2 * x)
    return 1.0 / 0.0;
  double xn = ((i & 1) == 0 ? -i / 2 : (-i - 1) / 2);
  i -= 4;
  *signgamp = ((i & 2) == 0 ? -1 : 1);

  SET_RESTORE_ROUND (FE_TONEAREST);

  /* Expand around the zero X0 = X0_HI + X0_LO.  */
  double x0_hi = lgamma_zeros[i][0], x0_lo = lgamma_zeros[i][1];
  double xdiff = x - x0_hi - x0_lo;

  /* For arguments in the range -3 to -2, use polynomial
     approximations to an adjusted version of the gamma function.  */
  if (i < 2)
    {
      int j = floor (-8 * x) - 16;
      double xm = (-33 - 2 * j) * 0.0625;
      double x_adj = x - xm;
      size_t deg = poly_deg[j];
      size_t end = poly_end[j];
      double g = poly_coeff[end];
      for (size_t j = 1; j <= deg; j++)
	g = g * x_adj + poly_coeff[end - j];
      return __log1p (g * xdiff / (x - xn));
    }

  /* The result we want is log (sinpi (X0) / sinpi (X))
     + log (gamma (1 - X0) / gamma (1 - X)).  */
  double x_idiff = fabs (xn - x), x0_idiff = fabs (xn - x0_hi - x0_lo);
  double log_sinpi_ratio;
  if (x0_idiff < x_idiff * 0.5)
    /* Use log not log1p to avoid inaccuracy from log1p of arguments
       close to -1.  */
    log_sinpi_ratio = __ieee754_log (lg_sinpi (x0_idiff)
				     / lg_sinpi (x_idiff));
  else
    {
      /* Use log1p not log to avoid inaccuracy from log of arguments
	 close to 1.  X0DIFF2 has positive sign if X0 is further from
	 XN than X is from XN, negative sign otherwise.  */
      double x0diff2 = ((i & 1) == 0 ? xdiff : -xdiff) * 0.5;
      double sx0d2 = lg_sinpi (x0diff2);
      double cx0d2 = lg_cospi (x0diff2);
      log_sinpi_ratio = __log1p (2 * sx0d2
				 * (-sx0d2 + cx0d2 * lg_cotpi (x_idiff)));
    }

  double log_gamma_ratio;
  double y0 = math_narrow_eval (1 - x0_hi);
  double y0_eps = -x0_hi + (1 - y0) - x0_lo;
  double y = math_narrow_eval (1 - x);
  double y_eps = -x + (1 - y);
  /* We now wish to compute LOG_GAMMA_RATIO
     = log (gamma (Y0 + Y0_EPS) / gamma (Y + Y_EPS)).  XDIFF
     accurately approximates the difference Y0 + Y0_EPS - Y -
     Y_EPS.  Use Stirling's approximation.  First, we may need to
     adjust into the range where Stirling's approximation is
     sufficiently accurate.  */
  double log_gamma_adj = 0;
  if (i < 6)
    {
      int n_up = (7 - i) / 2;
      double ny0, ny0_eps, ny, ny_eps;
      ny0 = math_narrow_eval (y0 + n_up);
      ny0_eps = y0 - (ny0 - n_up) + y0_eps;
      y0 = ny0;
      y0_eps = ny0_eps;
      ny = math_narrow_eval (y + n_up);
      ny_eps = y - (ny - n_up) + y_eps;
      y = ny;
      y_eps = ny_eps;
      double prodm1 = __lgamma_product (xdiff, y - n_up, y_eps, n_up);
      log_gamma_adj = -__log1p (prodm1);
    }
  double log_gamma_high
    = (xdiff * __log1p ((y0 - e_hi - e_lo + y0_eps) / e_hi)
       + (y - 0.5 + y_eps) * __log1p (xdiff / y) + log_gamma_adj);
  /* Compute the sum of (B_2k / 2k(2k-1))(Y0^-(2k-1) - Y^-(2k-1)).  */
  double y0r = 1 / y0, yr = 1 / y;
  double y0r2 = y0r * y0r, yr2 = yr * yr;
  double rdiff = -xdiff / (y * y0);
  double bterm[NCOEFF];
  double dlast = rdiff, elast = rdiff * yr * (yr + y0r);
  bterm[0] = dlast * lgamma_coeff[0];
  for (size_t j = 1; j < NCOEFF; j++)
    {
      double dnext = dlast * y0r2 + elast;
      double enext = elast * yr2;
      bterm[j] = dnext * lgamma_coeff[j];
      dlast = dnext;
      elast = enext;
    }
  double log_gamma_low = 0;
  for (size_t j = 0; j < NCOEFF; j++)
    log_gamma_low += bterm[NCOEFF - 1 - j];
  log_gamma_ratio = log_gamma_high + log_gamma_low;

  return log_sinpi_ratio + log_gamma_ratio;
}
