/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

ZMPLXFUNC_Z(__mth_i_cdlog)
{
  ZMPLXARGS_Z;
  double x, y;
  /*
  call libm's atan2 may cause ieee_invalid & ieee_overflow to
  be set (f19305)
  x = atan2(imag, real);
  Call our version, which for x64, is in rte/pgc/hammer/src-amd/datan2.c
  */
  x = __mth_i_datan2(imag, real);
  y = log(hypot(real, imag));
  ZRETURN_D_D(y, x);
}
