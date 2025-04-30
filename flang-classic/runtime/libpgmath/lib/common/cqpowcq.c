/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

#pragma clang diagnostic ignored "-Wunused-variable"
LZMPLXFUNC_LZ_LZ(__mth_i_cqpowcq)
{
  LZMPLXARGS_LZ_LZ;
  float128_t logr, logi, x, y, z, w;

  logr = logl(hypotl(real1, imag1));
  logi = atan2l(imag1, real1);

  x = expl(logr * real2 - logi * imag2);
  y = logr * imag2 + logi * real2;

  z = x * cosl(y);
  w = x * sinl(y);
  LZRETURN_Q_Q(z, w);
}
