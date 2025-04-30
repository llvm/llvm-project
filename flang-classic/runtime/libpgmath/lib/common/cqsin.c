/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

LZMPLXFUNC_LZ(__mth_i_cqsin)
{
  LZMPLXARGS_LZ;
  float128_t x, y;
  /*
  x = sinl(real) * coshl(imag);
  y = cosl(real) * sinhl(imag);
  */
  x = sinl(real);
  y = cosl(real);
  x = x * coshl(imag);
  y = y * sinhl(imag);
  LZRETURN_Q_Q(x, y);
}
