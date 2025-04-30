/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "mth_intrinsics.h"

vcs1_t
__gc_sqrt_1(vcs1_t x)
{
  return (__ZGVxN1v__mth_i_vc4(x, csqrtf));
}

vcs2_t
__gc_sqrt_2(vcs2_t x)
{
  return (__ZGVxN2v__mth_i_vc4(x, csqrtf));
}

double complex
__gz_sqrt_1(double complex x)
{
  return (csqrt(x));
}

vcd1_t
__gz_sqrt_1v(vcd1_t x)
{
  return (__ZGVxN1v__mth_i_vc8(x, csqrt));
}
