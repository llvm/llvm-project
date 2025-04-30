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
__gd_sin_2(vrd2_t x)
{
  return (__ZGVxN2v__mth_i_vr8(x, __mth_i_dsin));
}

vrd2_t
__gd_sin_2m(vrd2_t x, vid2_t mask)
{
  return (__ZGVxM2v__mth_i_vr8(x, mask, __mth_i_dsin));
}

double complex
__gz_sin_1(double complex x)
{
  return (csin(x));
}

vcd1_t
__gz_sin_1v(vcd1_t x)
{
  return (__ZGVxN1v__mth_i_vc8(x, csin));
}

#endif

