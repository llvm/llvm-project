/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mth_intrinsics.h"

#if defined(TARGET_LINUX_ARM64)
vrd2_t
__gd_pow_2(vrd2_t x, vrd2_t y)
{
  return (__ZGVxN2vv__mth_i_vr8vr8(x, y, __mth_i_dpowd));
}

vrd2_t
__gd_pow_2m(vrd2_t x, vrd2_t y, vid2_t mask)
{
  return (__ZGVxM2vv__mth_i_vr8vr8(x, y, mask, __mth_i_dpowd));
}

double complex
__gz_pow_1(double complex x, double complex y)
{
  return (cpow(x,y));
}

vcd1_t
__gz_pow_1v(vcd1_t x, vcd1_t y)
{
  return (__ZGVxN1vv__mth_i_vc8vc8(x, y, cpow));
}
#endif

