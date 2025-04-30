/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

#ifndef TARGET_WIN_X8664
double
__mth_i_dmod(double f, double g)
{
  return fmod(f, g);
}
#endif

double
__fmth_i_dmod(double f, double g) __attribute__ ((weak, alias ("__mth_i_dmod")));
