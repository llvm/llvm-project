/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdint.h>

int64_t
__mth_i_knint(float d)
{
  if (d > 0)
    return ((d < 8388608.f) ? (int64_t)(d + 0.5f) : (int64_t)(d));
  else
    return ((d > -8388608.f) ? (int64_t)(d - 0.5f) : (int64_t)(d));
}
