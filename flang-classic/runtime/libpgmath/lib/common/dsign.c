/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <math.h>

/* Assume little endian for now */
#define BITSDH(f) ((int *)&f)[1]
#define BITSDL(f) ((int *)&f)[0]
#define BSIGNF 0x80000000

double
__mth_i_dsign(double a, double b)
{
  double r;
  r = fabs(a);
  if (BITSDH(b) & BSIGNF) {
    /*r = -fabs(a);*/
    BITSDH(r) = BITSDH(r) | BSIGNF;
  }
  return r;
}
