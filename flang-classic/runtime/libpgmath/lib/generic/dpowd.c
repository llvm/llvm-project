/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"
double
__mth_i_dpowd(double x, double y)
{
  double f;
  /* f = __mth_i_pow(x, y); */
  f = pow(x, y);
  return f;
}
