/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* Intrinsic function which take quad precision arguments. */

#include "mthdecls.h"

float128_t
__mth_i_qpowk(float128_t x, long long i)
{
  long long k;
  float128_t f;

  f = 1;
  k = i;
  if (k < 0)
    k = -k;
  for (;;) {
    if (k & 1)
      f *= x;
    k >>= 1;
    if (k == 0)
      break;
    x *= x;
  }
  if (i < 0)
    f = 1.0 / f;
  return f;
}
