/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

float
__mth_i_rpowr(float arg1, float arg2)
{
  float f;
  f = powf(arg1, arg2);
  return f;
}
