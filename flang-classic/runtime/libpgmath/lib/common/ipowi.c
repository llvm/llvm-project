/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

int
__mth_i_ipowi(int x, int i)
{
  int f;

  /* special cases */

  if (x == 2) {
    if (i >= 0)
      return 1 << i;
    return 0;
  }
  if (i < 0) {
    if (x == 1)
      return 1;
    if (x == -1) {
      if (i & 1)
        return -1;
      return 1;
    }
    return 0;
  }

  if (i == 0)
    return 1;
  f = 1;
  while (1) {
    if (i & 1)
      f *= x;
    i >>= 1;
    if (i == 0)
      return f;
    x *= x;
  }
}
