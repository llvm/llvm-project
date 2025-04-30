/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* Intrinsic function which take quad precision arguments. */

#include "mthdecls.h"
/* INT */
#define LONG_LONG_MAX 9223372036854775807
long long __mth_i_kiqnint(float128_t d)
{
  if ((d > 0.0L) && (d > LONG_LONG_MAX))
    return LONG_LONG_MAX;

  if ((d < 0.0L) && (d < -LONG_LONG_MAX - 1LL))
    return -LONG_LONG_MAX - 1LL;

  return lroundl(d);
}
