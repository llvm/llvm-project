/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* Intrinsic function which take quad precision arguments. */

#include <math.h>
#include "mthdecls.h"

/* Assume little endian for now */
#define BITSDH(f) ((int *)&(f))[3]
#define BITSDL(f) ((int *)&(f))[0]
#define BSIGNF 0x80000000

float128_t
__mth_i_qsign(float128_t a, float128_t b)
{
  float128_t *pa = &a;
  float128_t *pb = &b;
  *pa = fabsl(a);
  if (is_little_endian()) {
    if (BITSDH(*pb) & BSIGNF) {
      BITSDH(*pa) |= BSIGNF;
    }
    return *pa;
  } else {
    if (BITSDL(*pb) & BSIGNF) {
      BITSDL(*pa) |= BSIGNF;
    }
    return *pa;
  }
}
