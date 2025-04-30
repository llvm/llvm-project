/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

ZMPLXFUNC_Z(__mth_i_cdsin)
{
  ZMPLXARGS_Z;
  double x, y;
  /*
  x = sin(real) * cosh(imag);
  y = cos(real) * sinh(imag);
  */
  x = sin(real);
  y = cos(real);
  x = x * cosh(imag);
  y = y * sinh(imag);
  ZRETURN_D_D(x, y);
}
