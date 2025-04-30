/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

CMPLXFUNC_C_F(__mth_i_cdivr)
{
  CMPLXARGS_C_F;
  float x, y;

  x = real / r;
  y = imag / r;
  CRETURN_F_F(x, y);
}
