/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <math.h>

#define BITSF(f) ((int *)&f)[0]
#define BSIGNF 0x80000000

float
__mth_i_sign(float a, float b)
{
  float r;
  r = fabsf(a);
  if (BITSF(b) & BSIGNF) {
    /*r = -fabsf(a);*/
    BITSF(r) = BITSF(r) | BSIGNF;
  }
  return r;
}
