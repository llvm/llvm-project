/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

CMPLXFUNC_C(__mth_i_csin)
{
  CMPLXARGS_C;
  float x, y;
  /*
  x = SINF(real) * COSHF(imag);
  y = COSF(real) * SINHF(imag);
  */
  x = sinf(real);
  y = cosf(real);
  x = x * coshf(imag);
  y = y * sinhf(imag);
  CRETURN_F_F(x, y);
}
