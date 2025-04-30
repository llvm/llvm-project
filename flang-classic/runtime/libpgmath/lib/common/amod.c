/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

#ifndef TARGET_WIN_X8664
float
__mth_i_amod(float f, float g)
{
  return fmodf(f, g);
}
#endif

float
__fmth_i_amod(float f, float g) __attribute__ ((weak, alias ("__mth_i_amod")));
