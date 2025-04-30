/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mth_intrinsics.h"

#if defined(TARGET_ARM64)
vrs4_t
__gs_pow_4(vrs4_t x, vrs4_t y)
{
  return (__ZGVxN4vv__mth_i_vr4vr4(x, y, __mth_i_rpowr));
}

vrs4_t
__gs_pow_4m(vrs4_t x, vrs4_t y, vis4_t mask)
{
  return (__ZGVxM4vv__mth_i_vr4vr4(x, y, mask, __mth_i_rpowr));
}

vcs1_t
__gc_pow_1(vcs1_t x, vcs1_t y)
{
  return (__ZGVxN1vv__mth_i_vc4vc4(x, y, cpowf));
}

vcs2_t
__gc_pow_2(vcs2_t x, vcs2_t y)
{
  return (__ZGVxN2vv__mth_i_vc4vc4(x, y, cpowf));
}
#endif

