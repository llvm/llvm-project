/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

ZMPLXFUNC_Z_K(__mth_i_cdpowk)
{
  ZMPLXARGS_Z_K;
  long long k;
  double fr, fi, gr, gi, tr, ti;
  static const double_complex_t c1plusi0 = PGMATH_CMPLX_CONST(1.0, 0.0);

  fr = 1;
  fi = 0;
  k = i;
  gr = real;
  gi = imag;
  if (k < 0)
    k = -k;
  while (k) {
    if (k & 1) {
      tr = fr * gr - fi * gi;
      ti = fr * gi + fi * gr;
      fr = tr;
      fi = ti;
    }
    k >>= 1;
    tr = gr * gr - gi * gi;
    ti = 2.0 * gr * gi;
    gr = tr;
    gi = ti;
  }

  double_complex_t z = pgmath_cmplx(fr, fi);
  if (i < 0) {
    ZMPLX_CALL_ZR_Z_Z(__mth_i_cddiv,z,c1plusi0,z);
  }
  ZRETURN_Z(z);

}
