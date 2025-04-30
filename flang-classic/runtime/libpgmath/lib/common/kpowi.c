/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

long long
__mth_i_kpowi(long long x, int i)
{
  int k;
  long long f;

  f = 1;
  k = i;
  if (k < 0)
    k = -k;
  for (;;) {
    if (k & 1)
      f *= x;
    k = (unsigned)k >> 1;
    if (k == 0)
      break;
    x *= x;
  }
  if (i < 0)
    f = 1.0f / f;
  return f;
}
