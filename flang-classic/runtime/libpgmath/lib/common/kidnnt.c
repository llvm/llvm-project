/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdint.h>

int64_t
__mth_i_kidnnt(double d)
{
  if (d > 0)
    return ((d < 4503599627370496.0) ? (int64_t)(d + 0.5) : (int64_t)(d));
  else
    return ((d > -4503599627370496.0) ? (int64_t)(d - 0.5) : (int64_t)(d));
}
