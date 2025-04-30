/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* Intrinsic function which take quad precision arguments. */

#include "mthdecls.h"
/* INT */
#define INT_MAX 2147483647
int __mth_i_iqnint(float128_t d)
{
  if ((d > 0.0L) && (d > INT_MAX))
    return INT_MAX;

  if ((d < 0.0L) && (d < -INT_MAX - 1))
    return -INT_MAX - 1;

  return lroundl(d);
}
