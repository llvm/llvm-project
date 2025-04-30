/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"

/*
 * Generic implementation of intrinsic sincos.
 *
 * Compiler expects two return values, thus using the complex type to implement
 * the return sequence.
 */

double_complex_t __mth_i_dsincos(double a)
{
  double s, c;
  __mth_dsincos(a, &s, &c);
  double_complex_t r = PGMATH_CMPLX_CONST(s, c);
  return r;
}
