/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

CMPLXFUNC_C(__mth_i_csqrt)
{
  CMPLXARGS_C;
  float a, x, y;

  a = hypotf(real, imag);
  if (a == 0) {
    x = 0;
    y = 0;
  } else if (real > 0) {
    x = sqrtf(0.5f * (a + real));
    y = 0.5 * (imag / x);
  } else {
    y = sqrtf(0.5f * (a - real));
    y = copysignf(y,imag);
    x = 0.5f * (imag / y);
  }
  CRETURN_F_F(x, y);
}
