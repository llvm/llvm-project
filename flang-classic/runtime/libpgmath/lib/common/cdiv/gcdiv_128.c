/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "mth_intrinsics.h"

vcs1_t
__gc_div_1(vcs1_t x, vcs1_t y)
{
  return (__ZGVxN1vv__mth_i_vc4vc4(x, y, __mth_i_cdiv_c99));
}

vcs2_t
__gc_div_2(vcs2_t x, vcs2_t y)
{
  return (__ZGVxN2vv__mth_i_vc4vc4(x, y, __mth_i_cdiv_c99));
}

double complex
__gz_div_1(double complex x, double complex y)
{
  return (__mth_i_cddiv_c99(x, y));
}

vcd1_t
__gz_div_1v(vcd1_t x, vcd1_t y)
{
  return (__ZGVxN1vv__mth_i_vc8vc8(x, y, __mth_i_cddiv_c99));
}
