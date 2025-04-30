/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mth_intrinsics.h"

#if defined(TARGET_LINUX_ARM64)
vrd2_t
__gd_log_2(vrd2_t x)
{
  return (__ZGVxN2v__mth_i_vr8(x, __mth_i_dlog));
}

vrd2_t
__gd_log_2m(vrd2_t x, vid2_t mask)
{
  return (__ZGVxM2v__mth_i_vr8(x, mask, __mth_i_dlog));
}

double complex
__gz_log_1(double complex x)
{
  return (clog(x));
}

#define clog    __mth_i_cdlog_c99

vcd1_t
__gz_log_1v(vcd1_t x)
{
  return (__ZGVxN1v__mth_i_vc8(x, clog));
}
#endif

