/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* Intrinsic function which take quad precision arguments. */

#include <stdint.h>
#include "mthdecls.h"

typedef union {
  float128_t  f;
  __uint128_t i;
} FPI;

const __int128_t BITS1 = 0x7fff000000000000;
const __int128_t BITS2 = 0x8000000000000000;
#define EXPBIAS 16383
#define MANTBITS 112
#define SIGN_MASK (BITS2 << 64)
#define GET_EXP(u) (__int128_t)(((u) & (BITS1 << 64)) >> MANTBITS)

float128_t __mth_i_qint(float128_t xx)
{
  __int128_t xexp;
  __uint128_t ux, mask;
  float128_t x;
  FPI fpi;

  x = xx;
  fpi.f = x;
  ux = fpi.i;
  xexp = GET_EXP(ux) - EXPBIAS;
  if (xexp < 0) {
    /* |x| < 0  =>  zero with the original sign */
    fpi.i = (ux & SIGN_MASK);
  } else if (xexp < MANTBITS) {
    /* 1 <= |x| < 2^113:
     * just mask out the trailing bits of the mantiassa beyond the
     * range of the exponent; mask out the exponent field as well.
     */
    mask = ((__uint128_t)1 << (MANTBITS - xexp)) - 1;
    fpi.i = ux & ~mask;
  }

  /* else illegal input, nan, inf, overflow, ...; just return it */
  return fpi.f;
}
