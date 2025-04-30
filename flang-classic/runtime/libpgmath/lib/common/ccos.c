/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

CMPLXFUNC_C(__mth_i_ccos)
{
  CMPLXARGS_C;
  float x, y;
  /*
  x = COSF(real) * COSHF(imag);
  y = -SINF(real) * SINHF(imag);
  */
  x = cosf(real);
  y = sinf(real);
  x = x * coshf(imag);
  y = -y * sinhf(imag);
  CRETURN_F_F(x, y);
}
