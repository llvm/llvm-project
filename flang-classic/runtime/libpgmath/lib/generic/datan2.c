/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#if !defined(_WIN64)
#include "mthdecls.h"
#else
double atan2(double x, double y);
#endif
double
__mth_i_datan2(double x, double y)
{
  return atan2(x, y);
}
