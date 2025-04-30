/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdint.h>
#include "mthdecls.h"

#if     defined(TARGET_LINUX_POWER)
double
__mth_i_dint(double x)
{
  double d;
  asm("friz %0, %1"
     : "=d"(d)
     : "d"(x)
     :
     );
  return d;
}
#else   /* defined(TARGET_LINUX_POWER) */
typedef union {
  double f;
  uint64_t i;
} FPI;

#define EXPBIAS 1023
#define MANTBITS 52
#define GET_EXP(u) (int64_t)(((u)&0x7ff0000000000000) >> MANTBITS)

double
__mth_i_dint(double xx)
{
  int64_t xexp;
  uint64_t ux, mask;
  double x;
  FPI fpi;

  x = xx;
  fpi.f = x;
  ux = fpi.i;
  xexp = GET_EXP(ux) - EXPBIAS;
  if (xexp < 0) {
    /* |x| < 0  =>  zero with the original sign */
    fpi.i = (ux & 0x8000000000000000);
  } else if (xexp < MANTBITS) {
    /* 1 <= |x| < 2^53:
     *    just mask out the trailing bits of the mantiassa beyond the
     *    range of the exponent; mask out the exponent field as well.
     */
    mask = ((uint64_t)1 << (MANTBITS - xexp)) - 1;
    fpi.i = ux & ~mask;
  }
  /* else illegal input, nan, inf, overflow, ...; just return it */

  return fpi.f;
}
#endif  /* defined(TARGET_LINUX_POWER) */
