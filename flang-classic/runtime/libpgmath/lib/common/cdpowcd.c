/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

ZMPLXFUNC_Z_Z(__mth_i_cdpowcd)
{
  ZMPLXARGS_Z_Z;
  double logr, logi, x, y, z, w;

  logr = log(hypot(real1, imag1));
  logi = atan2(imag1, real1);

  x = exp(logr * real2 - logi * imag2);
  y = logr * imag2 + logi * real2;

  z = x * cos(y);
  w = x * sin(y);
  ZRETURN_D_D(z, w);
}
