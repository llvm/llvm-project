/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

ZMPLXFUNC_Z(__mth_i_cdsqrt)
{
  ZMPLXARGS_Z;
  double a, x, y;

  a = hypot(real, imag);
  if (a == 0) {
    x = 0;
    y = 0;
  } else if (real > 0) {
    x = sqrt(0.5 * (a + real));
    y = 0.5 * (imag / x);
  } else {
    y = sqrt(0.5 * (a - real));
    y = copysign(y,imag);
    x = 0.5 * (imag / y);
  }
  ZRETURN_D_D(x, y);
}
