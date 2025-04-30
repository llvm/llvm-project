/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "mthdecls.h"

/*
 * libm's roundf() could also be used if compiled with:
 *      _ISOC99_SOURCE || _POSIX_C_SOURCE >= 200112L
 */

#if     defined(TARGET_LINUX_POWER)
float
__mth_i_anint(float f)
{
  float x;
  asm("frin %0, %1"
     : "=d"(x)
     : "d"(f)
     :
     );
  return x;
}

#elif   defined(__aarch64__)
float
__mth_i_anint(float f)
{
  float r;
  asm("frinta   %s0, %s1"
    : "=w"(r)
    : "w"(f)
    :);
  return r;
}

#else
#include <math.h>
#include <ieee754.h>

float
__mth_i_anint(float f)
{
  float x = f;     /* Cases where f == 0.0 or f == NaN */
  union ieee754_float *u = (union ieee754_float *)&x;

  /*
   * Depending on the default rounding mode of the processor, the logic
   * below with modff(f + 0.5f) can result in a bogus rounding when 0.5f
   * is normalized such that it falls within the guard or round bits.
   *
   * Fast return if the exponent guarantees that the floating point number
   * is a whole integer.
   *
   * This quick exit also catches infinities and NaNs.
   */

  if (u->ieee.exponent >= IEEE754_FLOAT_BIAS+23) return x;

  if (f > 0.0)
    (void)modff(f + 0.5f, &x);
  else if (f < 0.0)
    (void)modff(f - 0.5f, &x);
  return x;
}
#endif  /* #if     defined(TARGET_LINUX_POWER) */
