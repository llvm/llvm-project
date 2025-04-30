/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifdef  TARGET_X8664
#error  Single precision - generic sincos() will not work on X86-64 systems.
#endif

#include "mthdecls.h"

/*
 * Generic implementation of intrinsic sincos.
 *
 * Compiler expects two return values, thus using the complex type to implement
 * the return sequence.
 */

float_complex_t __mth_i_sincos(float a)
{
  float s, c;
  __mth_sincos(a, &s, &c);
  float_complex_t r = PGMATH_CMPLX_CONST(s, c);
  return r;
}
