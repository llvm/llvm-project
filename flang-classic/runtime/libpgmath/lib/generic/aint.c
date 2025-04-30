/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#if     defined(TARGET_LINUX_POWER)
float
__mth_i_aint(float x)
{
  float f;
  asm("friz %0, %1"
         : "=d"(f)
         : "d"(x)
         :
         );
  return f;
}
#else   /* #if     defined(TARGET_POWER) */
typedef union {
  float f;
  unsigned int i;
} FPI;

#define EXPBIAS 127
#define MANTBITS 23
#define GET_EXP(u) (int)(((u)&0x7fffffff) >> MANTBITS)

float
__mth_i_aint(float xx)
{
  unsigned int ux, mask;
  int xexp;
  float x;
  FPI fpi;

  x = xx;
  fpi.f = x;
  ux = fpi.i;

  xexp = GET_EXP(ux) - EXPBIAS;
  if (xexp < 0) {
    /* |x| < 0 => zero with the original sign */
    fpi.i = ux & 0x80000000;
  } else if (xexp < MANTBITS) {
    /* 1 <= |x| < 2^24:
     *    just mask out the trailing bits of the mantissa beyond the
     *    range of the exponent; mask out the exponent field as well.
     */
    mask = (1 << (MANTBITS - xexp)) - 1;
    fpi.i = ux & ~mask;
  }
  /* else for illegal input, nan, inf, overflow, ...; just return it */

  return fpi.f;
}
#endif  /* #if     defined(TARGET_POWER) */
