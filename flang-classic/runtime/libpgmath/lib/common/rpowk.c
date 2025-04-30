/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

/* should handle 0**0 w/ exception */
float
__mth_i_rpowk(float x, long long i)
{
  long long k;
  float f;

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
    f = 1.0f / f;
  return f;
}
