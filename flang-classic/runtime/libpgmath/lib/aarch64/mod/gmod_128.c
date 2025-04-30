/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mth_intrinsics.h"

#if defined(TARGET_ARM64)
vrs4_t
__gs_mod_4(vrs4_t x, vrs4_t y)
{
  return (__ZGVxN4vv__mth_i_vr4vr4(x, y, __mth_i_amod));
}

vrs4_t
__gs_mod_4m(vrs4_t x, vrs4_t y, vis4_t mask)
{
  return (__ZGVxM4vv__mth_i_vr4vr4(x, y, mask, __mth_i_amod));
}

vrd2_t
__gd_mod_2(vrd2_t x, vrd2_t y)
{
  return (__ZGVxN2vv__mth_i_vr8vr8(x, y, __mth_i_dmod));
}

vrd2_t
__gd_mod_2m(vrd2_t x, vrd2_t y, vid2_t mask)
{
  return (__ZGVxM2vv__mth_i_vr8vr8(x, y, mask, __mth_i_dmod));
}
#endif

