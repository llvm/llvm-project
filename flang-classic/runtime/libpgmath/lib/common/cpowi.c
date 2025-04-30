/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

CMPLXFUNC_C_I(__mth_i_cpowi)
{
  CMPLXARGS_C_I;
  int k;
  float fr, fi, gr, gi, tr, ti;
  float_complex_t c;
  static const float_complex_t c1plusi0 = PGMATH_CMPLX_CONST(1.0, 0.0);

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
    k = (unsigned)k >> 1;
    tr = gr * gr - gi * gi;
    ti = 2.0 * gr * gi;
    gr = tr;
    gi = ti;
  }

  c = pgmath_cmplxf(fr, fi);
  if (i < 0) {
    CMPLX_CALL_CR_C_C(__mth_i_cdiv,c,c1plusi0,c);
  }
  CRETURN_C(c);
}
