/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdint.h>
#include "mthdecls.h"
#define MYPI 180.0
#define HFPI 90.0

double
__mth_i_dtand(double d)
{
  union {
    double d;
    uint64_t t;
  } u;
  if ((fmod(fabs(d), MYPI) - HFPI) == 0.0) {
    u.t = 0x7ff0000000000000;
    return u.d;
  }
  return (tan(CNVRTDEG(d)));
}
