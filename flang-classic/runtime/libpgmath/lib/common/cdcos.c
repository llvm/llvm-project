/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

ZMPLXFUNC_Z(__mth_i_cdcos)
{
  ZMPLXARGS_Z;
  double x, y;
  /*
  x = cos(real) * cosh(imag);
  y = -sin(real) * sinh(imag);
  */
  // x = cos(real);
  // y = sin(real);
  __mth_dsincos(real, &y, &x);
  x = x * cosh(imag);
  y = -y * sinh(imag);
  ZRETURN_D_D(x, y);
}
