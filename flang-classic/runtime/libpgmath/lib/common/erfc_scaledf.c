/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* inhibit floating point copy propagation */
#pragma global - Mx, 6, 0x100

#include "mthdecls.h"
#include <float.h>

#define pi 3.1415926535897932384626434f
#define sqrtpi 1.77245385090551602729f
#define epsr4 2.2204460492503131e-016f
#define xinfr4 FLT_MAX
#define xminr4 FLT_MIN
#define xsmallr4 epsr4 / 2.0f
#define xmaxr4 1.0f / (sqrtpi * xminr4)

/*  mathematical constants */
#define zero 0.0e0f
#define four 4.0e0f
#define one 1.0e0f
#define half 0.5e0f
#define two 2.0e0f
#define sqrpi 5.6418958354775628695e-1f
#define thresh 0.46875e0f
#define sixten 16.0e0f

/*  machine-dependent constants: ieee float precision values */
#define xneg -26.628e0f
#define xbig 26.543e0f
#define xhuge 6.71e7f

/*  coefficients for approximation to  erf  in first interval */

static float a[5] = {3.16112374387056560e00f, 1.13864154151050156e02f,
                     3.77485237685302021e02f, 3.20937758913846947e03f,
                     1.85777706184603153e-1f};
static float b[4] = {2.36012909523441209e01f, 2.44024637934444173e02f,
                     1.28261652607737228e03f, 2.84423683343917062e03f};

/*  coefficients for approximation to  erfc  in second interval */
static float c[9] = {
    5.64188496988670089e-1f, 8.88314979438837594e00f, 6.61191906371416295e01f,
    2.98635138197400131e02f, 8.81952221241769090e02f, 1.71204761263407058e03f,
    2.05107837782607147e03f, 1.23033935479799725e03f, 2.15311535474403846e-8f};
static float d[8] = {1.57449261107098347e01f, 1.17693950891312499e02f,
                     5.37181101862009858e02f, 1.62138957456669019e03f,
                     3.29079923573345963e03f, 4.36261909014324716e03f,
                     3.43936767414372164e03f, 1.23033935480374942e03f};

/*  coefficients for approximation to  erfc  in third interval */
static float p[6] = {3.05326634961232344e-1f, 3.60344899949804439e-1f,
                     1.25781726111229246e-1f, 1.60837851487422766e-2f,
                     6.58749161529837803e-4f, 1.63153871373020978e-2f};
static float q[5] = {2.56852019228982242e00f, 1.87295284992346047e00f,
                     5.27905102951428412e-1f, 6.05183413124413191e-2f,
                     2.33520497626869185e-3f};

float
__mth_i_erfc_scaled(float arg)
{
  float x, y, ysq, xnum, xden, del;
  int i;
  float result;

  x = arg;
  y = fabs(x);

  if (y <= thresh) {
    /* evaluate  erf  for  |x| <= 0.46875 */
    ysq = zero;
    if (y > xsmallr4)
      ysq = y * y;
    xnum = a[4] * ysq;
    xden = ysq;
    for (i = 0; i < 3; i++) {
      xnum = (xnum + a[i]) * ysq;
      xden = (xden + b[i]) * ysq;
    }
    result = x * (xnum + a[3]) / (xden + b[3]);
    result = one - result;
    result = exp(ysq) * result;
    goto ret;
  } else if (y <= four) {
    /*  evaluate  erfc  for 0.46875 <= |x| <= 4.0 */
    xnum = c[8] * y;
    xden = y;
    for (i = 0; i < 7; i++) {
      xnum = (xnum + c[i]) * y;
      xden = (xden + d[i]) * y;
    }
    result = (xnum + c[7]) / (xden + d[7]);
  } else {
    /*  evaluate  erfc  for |x| > 4.0 */
    result = zero;
    if (y >= xbig) {
      if (y > xmaxr4)
        goto negval;
      if (y >= xhuge) {
        result = sqrpi / y;
        goto negval;
      }
    }
    ysq = one / (y * y);
    xnum = p[5] * ysq;
    xden = ysq;
    for (i = 0; i < 4; i++) {
      xnum = (xnum + p[i]) * ysq;
      xden = (xden + q[i]) * ysq;
    }
    result = ysq * (xnum + p[4]) / (xden + q[4]);
    result = (sqrpi - result) / y;
  }
negval:
  /*  fix up for negative argument, erf, etc. */
  if (x < zero) {
    if (x < xneg) {
      result = xinfr4;
    } else {
#if defined(TARGET_WIN)
      double tmp = x * sixten;
      long l = tmp;
      tmp = l;
      ysq = tmp / sixten;
#else
      ysq = trunc(x * sixten) / sixten;
#endif
      del = (x - ysq) * (x + ysq);
      y = exp(ysq * ysq) * exp(del);
      result = (y + y) - result;
    }
  }
ret:
  return result;
}
