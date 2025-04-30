/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "float128.h"

extern long double remainderl(long double, long double);

float128_t __mth_i_qremainder(float128_t x, float128_t y)
{
  return remainderl(x, y);
}

