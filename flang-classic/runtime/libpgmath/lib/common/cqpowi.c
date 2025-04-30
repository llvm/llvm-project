/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

LZMPLXFUNC_LZ_I(__mth_i_cqpowi)
{
  LZMPLXARGS_LZ_I;
  int k;
  float128_t fr, fi, gr, gi, tr, ti;
  static const quad_complex_t c1plusi0 = PGMATH_CMPLX_CONST(1.0, 0.0);

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

  quad_complex_t lz = pgmath_cmplxl(fr, fi);
  if (i < 0) {
    LZMPLX_CALL_LZR_LZ_LZ(__mth_i_cqdiv, lz, c1plusi0, lz);
  }
  LZRETURN_LZ(lz);
}
