/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#if defined(TARGET_ARM64)
#include "mth_intrinsics.h"

vrs4_t
__gs_powi1_4(vrs4_t x, int32_t iy)
{
  return(__ZGVxN4v__mth_i_vr4si4(x, iy, __mth_i_rpowi));
}

vrs4_t
__gs_powi1_4m(vrs4_t x, int32_t iy, vis4_t mask)
{
  return(__ZGVxM4v__mth_i_vr4si4(x, iy, mask, __mth_i_rpowi));
}

vrs4_t
__gs_powi_4(vrs4_t x, vis4_t iy)
{
  return(__ZGVxN4vv__mth_i_vr4vi4(x, iy, __mth_i_rpowi));
}

vrs4_t
__gs_powi_4m(vrs4_t x, vis4_t iy, vis4_t mask)
{
  return(__ZGVxM4vv__mth_i_vr4vi4(x, iy, mask, __mth_i_rpowi));
}

vrs4_t
__gs_powk1_4(vrs4_t x, long long iy)
{
  return(__ZGVxN4v__mth_i_vr4si8(x, iy, __mth_i_rpowk));
}

vrs4_t
__gs_powk1_4m(vrs4_t x, long long iy, vis4_t mask)
{
  return(__ZGVxM4v__mth_i_vr4si8(x, iy, mask, __mth_i_rpowk));
}

vrs4_t
__gs_powk_4(vrs4_t x, vid2_t iyl, vid2_t iyu)
{
  return(__ZGVxN4vv__mth_i_vr4vi8(x, iyl, iyu, __mth_i_rpowk));
}

vrs4_t
__gs_powk_4m(vrs4_t x, vid2_t iyl, vid2_t iyu, vis4_t mask)
{
  return(__ZGVxM4vv__mth_i_vr4vi8(x, iyl, iyu, mask, __mth_i_rpowk));
}
#endif

