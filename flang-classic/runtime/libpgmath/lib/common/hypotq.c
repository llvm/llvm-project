/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

float128_t
__mth_i_qhypot(float128_t x, float128_t y)
{
  float128_t f = hypotl(x, y);
  return f;
}
