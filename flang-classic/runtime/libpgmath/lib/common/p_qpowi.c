/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

/* should handle 0**0 w/ exception */

/*
 * __pmth_i_qpowk(float128_t x, long long iy)
 *
 * Return R16**I8 with intermediate terms computed as R16.
 * Most likely used with -Kieee (precise).
 */

float128_t
__pmth_i_qpowk(float128_t x16, long long i8)
{
  long long k;
  float128_t r16;

  r16 = 1.0;
  k = i8;
  if (k < 0)
    k = -k;
  for (;;) {
    if (k & 1)
      r16 *= x16;
    k >>= 1;
    if (k == 0)
      break;
    x16 *= x16;
  }
  if (i8 < 0)
    r16 = 1.0 / r16;
  return r16;
}

/*
 * __pmth_i_qpowi(float128_t x, int i4)
 * Return R16**I4 with intermediate terms computed as r16.
 *
 * Most likely used with -Kieee (precise).
 */

float128_t
__pmth_i_qpowi(float128_t x16, int i4)
{
  return __pmth_i_qpowk(x16, i4);
}
