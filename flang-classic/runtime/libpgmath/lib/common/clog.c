/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

CMPLXFUNC_C(__mth_i_clog)
{
  CMPLXARGS_C;
  float x, y;
  x = atan2f(imag, real);
  y = logf(hypotf(real, imag));
  CRETURN_F_F(y, x);
}
