/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

ZMPLXFUNC_Z_Z(__mth_i_cddiv)
{
  ZMPLXARGS_Z_Z;
  double x, y;
  double r, d, r_mag, i_mag;

  r_mag = real2;
  if (r_mag < 0)
    r_mag = -r_mag;
  i_mag = imag2;
  if (i_mag < 0)
    i_mag = -i_mag;
  /* avoid overflow */
  if (r_mag <= i_mag) {
    r = real2 / imag2;
    d = 1.0 / (imag2 * (1 + r * r));
    x = (real1 * r + imag1) * d;
    y = (imag1 * r - real1) * d;
  } else {
    r = imag2 / real2;
    d = 1.0 / (real2 * (1 + r * r));
    x = (real1 + imag1 * r) * d;
    y = (imag1 - real1 * r) * d;
  }
  ZRETURN_D_D(x, y);
}
