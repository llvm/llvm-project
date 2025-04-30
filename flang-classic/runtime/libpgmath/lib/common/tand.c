/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mthdecls.h"
#define MYPI 180.0
#define HFPI 90.0

float
__mth_i_tand(float f)
{
  union {
    float f;
    unsigned int i;
  } u;
  if ((fmodf(fabsf(f), MYPI) - HFPI) == 0.0) {
    u.i = 0x7f800000;
    return u.f;
  }
  return tanf(CNVRTDEG(f));
}
