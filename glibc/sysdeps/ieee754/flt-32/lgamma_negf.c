/* lgammaf expanding around zeros.
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

static const float lgamma_zeros[][2] =
  {
    { -0x2.74ff94p+0f, 0x1.3fe0f2p-24f },
    { -0x2.bf682p+0f, -0x1.437b2p-24f },
    { -0x3.24c1b8p+0f, 0x6.c34cap-28f },
    { -0x3.f48e2cp+0f, 0x1.707a04p-24f },
    { -0x4.0a13ap+0f, 0x1.e99aap-24f },
    { -0x4.fdd5ep+0f, 0x1.64454p-24f },
    { -0x5.021a98p+0f, 0x2.03d248p-24f },
    { -0x5.ffa4cp+0f, 0x2.9b82fcp-24f },
    { -0x6.005ac8p+0f, -0x1.625f24p-24f },
    { -0x6.fff3p+0f, 0x2.251e44p-24f },
    { -0x7.000dp+0f, 0x8.48078p-28f },
    { -0x7.fffe6p+0f, 0x1.fa98c4p-28f },
    { -0x8.0001ap+0f, -0x1.459fcap-28f },
    { -0x8.ffffdp+0f, -0x1.c425e8p-24f },
    { -0x9.00003p+0f, 0x1.c44b82p-24f },
    { -0xap+0f, 0x4.9f942p-24f },
    { -0xap+0f, -0x4.9f93b8p-24f },
    { -0xbp+0f, 0x6.b9916p-28f },
    { -0xbp+0f, -0x6.b9915p-28f },
    { -0xcp+0f, 0x8.f76c8p-32f },
    { -0xcp+0f, -0x8.f76c7p-32f },
    { -0xdp+0f, 0xb.09231p-36f },
    { -0xdp+0f, -0xb.09231p-36f },
    { -0xep+0f, 0xc.9cba5p-40f },
    { -0xep+0f, -0xc.9cba5p-40f },
    { -0xfp+0f, 0xd.73f9fp-44f },
  };

static const float e_hi = 0x2.b7e15p+0f, e_lo = 0x1.628aeep-24f;

/* Coefficients B_2k / 2k(2k-1) of x^-(2k-1) in Stirling's
   approximation to lgamma function.  */

static const float lgamma_coeff[] =
  {
    0x1.555556p-4f,
    -0xb.60b61p-12f,
    0x3.403404p-12f,
  };

#define NCOEFF (sizeof (lgamma_coeff) / sizeof (lgamma_coeff[0]))

/* Polynomial approximations to (|gamma(x)|-1)(x-n)/(x-x0), where n is
   the integer end-point of the half-integer interval containing x and
   x0 is the zero of lgamma in that half-integer interval.  Each
   polynomial is expressed in terms of x-xm, where xm is the midpoint
   of the interval for which the polynomial applies.  */

static const float poly_coeff[] =
  {
    /* Interval [-2.125, -2] (polynomial degree 5).  */
    -0x1.0b71c6p+0f,
    -0xc.73a1ep-4f,
    -0x1.ec8462p-4f,
    -0xe.37b93p-4f,
    -0x1.02ed36p-4f,
    -0xe.cbe26p-4f,
    /* Interval [-2.25, -2.125] (polynomial degree 5).  */
    -0xf.29309p-4f,
    -0xc.a5cfep-4f,
    0x3.9c93fcp-4f,
    -0x1.02a2fp+0f,
    0x9.896bep-4f,
    -0x1.519704p+0f,
    /* Interval [-2.375, -2.25] (polynomial degree 5).  */
    -0xd.7d28dp-4f,
    -0xe.6964cp-4f,
    0xb.0d4f1p-4f,
    -0x1.9240aep+0f,
    0x1.dadabap+0f,
    -0x3.1778c4p+0f,
    /* Interval [-2.5, -2.375] (polynomial degree 6).  */
    -0xb.74ea2p-4f,
    -0x1.2a82cp+0f,
    0x1.880234p+0f,
    -0x3.320c4p+0f,
    0x5.572a38p+0f,
    -0x9.f92bap+0f,
    0x1.1c347ep+4f,
    /* Interval [-2.625, -2.5] (polynomial degree 6).  */
    -0x3.d10108p-4f,
    0x1.cd5584p+0f,
    0x3.819c24p+0f,
    0x6.84cbb8p+0f,
    0xb.bf269p+0f,
    0x1.57fb12p+4f,
    0x2.7b9854p+4f,
    /* Interval [-2.75, -2.625] (polynomial degree 6).  */
    -0x6.b5d25p-4f,
    0x1.28d604p+0f,
    0x1.db6526p+0f,
    0x2.e20b38p+0f,
    0x4.44c378p+0f,
    0x6.62a08p+0f,
    0x9.6db3ap+0f,
    /* Interval [-2.875, -2.75] (polynomial degree 5).  */
    -0x8.a41b2p-4f,
    0xc.da87fp-4f,
    0x1.147312p+0f,
    0x1.7617dap+0f,
    0x1.d6c13p+0f,
    0x2.57a358p+0f,
    /* Interval [-3, -2.875] (polynomial degree 5).  */
    -0xa.046d6p-4f,
    0x9.70b89p-4f,
    0xa.a89a6p-4f,
    0xd.2f2d8p-4f,
    0xd.e32b4p-4f,
    0xf.fb741p-4f,
  };

static const size_t poly_deg[] =
  {
    5,
    5,
    5,
    6,
    6,
    6,
    5,
    5,
  };

static const size_t poly_end[] =
  {
    5,
    11,
    17,
    24,
    31,
    38,
    44,
    50,
  };

/* Compute sin (pi * X) for -0.25 <= X <= 0.5.  */

static float
lg_sinpi (float x)
{
  if (x <= 0.25f)
    return __sinf ((float) M_PI * x);
  else
    return __cosf ((float) M_PI * (0.5f - x));
}

/* Compute cos (pi * X) for -0.25 <= X <= 0.5.  */

static float
lg_cospi (float x)
{
  if (x <= 0.25f)
    return __cosf ((float) M_PI * x);
  else
    return __sinf ((float) M_PI * (0.5f - x));
}

/* Compute cot (pi * X) for -0.25 <= X <= 0.5.  */

static float
lg_cotpi (float x)
{
  return lg_cospi (x) / lg_sinpi (x);
}

/* Compute lgamma of a negative argument -15 < X < -2, setting
   *SIGNGAMP accordingly.  */

float
__lgamma_negf (float x, int *signgamp)
{
  /* Determine the half-integer region X lies in, handle exact
     integers and determine the sign of the result.  */
  int i = floorf (-2 * x);
  if ((i & 1) == 0 && i == -2 * x)
    return 1.0f / 0.0f;
  float xn = ((i & 1) == 0 ? -i / 2 : (-i - 1) / 2);
  i -= 4;
  *signgamp = ((i & 2) == 0 ? -1 : 1);

  SET_RESTORE_ROUNDF (FE_TONEAREST);

  /* Expand around the zero X0 = X0_HI + X0_LO.  */
  float x0_hi = lgamma_zeros[i][0], x0_lo = lgamma_zeros[i][1];
  float xdiff = x - x0_hi - x0_lo;

  /* For arguments in the range -3 to -2, use polynomial
     approximations to an adjusted version of the gamma function.  */
  if (i < 2)
    {
      int j = floorf (-8 * x) - 16;
      float xm = (-33 - 2 * j) * 0.0625f;
      float x_adj = x - xm;
      size_t deg = poly_deg[j];
      size_t end = poly_end[j];
      float g = poly_coeff[end];
      for (size_t j = 1; j <= deg; j++)
	g = g * x_adj + poly_coeff[end - j];
      return __log1pf (g * xdiff / (x - xn));
    }

  /* The result we want is log (sinpi (X0) / sinpi (X))
     + log (gamma (1 - X0) / gamma (1 - X)).  */
  float x_idiff = fabsf (xn - x), x0_idiff = fabsf (xn - x0_hi - x0_lo);
  float log_sinpi_ratio;
  if (x0_idiff < x_idiff * 0.5f)
    /* Use log not log1p to avoid inaccuracy from log1p of arguments
       close to -1.  */
    log_sinpi_ratio = __ieee754_logf (lg_sinpi (x0_idiff)
				      / lg_sinpi (x_idiff));
  else
    {
      /* Use log1p not log to avoid inaccuracy from log of arguments
	 close to 1.  X0DIFF2 has positive sign if X0 is further from
	 XN than X is from XN, negative sign otherwise.  */
      float x0diff2 = ((i & 1) == 0 ? xdiff : -xdiff) * 0.5f;
      float sx0d2 = lg_sinpi (x0diff2);
      float cx0d2 = lg_cospi (x0diff2);
      log_sinpi_ratio = __log1pf (2 * sx0d2
				  * (-sx0d2 + cx0d2 * lg_cotpi (x_idiff)));
    }

  float log_gamma_ratio;
  float y0 = math_narrow_eval (1 - x0_hi);
  float y0_eps = -x0_hi + (1 - y0) - x0_lo;
  float y = math_narrow_eval (1 - x);
  float y_eps = -x + (1 - y);
  /* We now wish to compute LOG_GAMMA_RATIO
     = log (gamma (Y0 + Y0_EPS) / gamma (Y + Y_EPS)).  XDIFF
     accurately approximates the difference Y0 + Y0_EPS - Y -
     Y_EPS.  Use Stirling's approximation.  */
  float log_gamma_high
    = (xdiff * __log1pf ((y0 - e_hi - e_lo + y0_eps) / e_hi)
       + (y - 0.5f + y_eps) * __log1pf (xdiff / y));
  /* Compute the sum of (B_2k / 2k(2k-1))(Y0^-(2k-1) - Y^-(2k-1)).  */
  float y0r = 1 / y0, yr = 1 / y;
  float y0r2 = y0r * y0r, yr2 = yr * yr;
  float rdiff = -xdiff / (y * y0);
  float bterm[NCOEFF];
  float dlast = rdiff, elast = rdiff * yr * (yr + y0r);
  bterm[0] = dlast * lgamma_coeff[0];
  for (size_t j = 1; j < NCOEFF; j++)
    {
      float dnext = dlast * y0r2 + elast;
      float enext = elast * yr2;
      bterm[j] = dnext * lgamma_coeff[j];
      dlast = dnext;
      elast = enext;
    }
  float log_gamma_low = 0;
  for (size_t j = 0; j < NCOEFF; j++)
    log_gamma_low += bterm[NCOEFF - 1 - j];
  log_gamma_ratio = log_gamma_high + log_gamma_low;

  return log_sinpi_ratio + log_gamma_ratio;
}
