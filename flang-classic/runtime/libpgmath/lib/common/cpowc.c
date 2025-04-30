/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

CMPLXFUNC_C_C(__mth_i_cpowc)
{
  CMPLXARGS_C_C;
  float logr, logi, x, y, z, w;

  logr = logf(hypotf(real1, imag1));
  logi = atan2f(imag1, real1);

  x = expf(logr * real2 - logi * imag2);
  y = logr * imag2 + logi * real2;

  z = x * cosf(y);
  w = x * sinf(y);
  CRETURN_F_F(z, w);
}
