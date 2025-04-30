/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

// TODO: Implement for TARGET_WIN_ARM64
#if defined(TARGET_LINUX_ARM64)

#include "mth_intrinsics.h"

vrd2_t
__gd_atan_2(vrd2_t x)
{
  return (__ZGVxN2v__mth_i_vr8(x, __mth_i_datan));
}

vrd2_t
__gd_atan_2m(vrd2_t x, vid2_t mask)
{
  return (__ZGVxM2v__mth_i_vr8(x, mask, __mth_i_datan));
}

double complex
__gz_atan_1(double complex x)
{
  return (catan(x));
}

vcd1_t
__gz_atan_1v(vcd1_t x)
{
  return (__ZGVxN1v__mth_i_vc8(x, catan));
}
#endif

