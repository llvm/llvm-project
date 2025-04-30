/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mth_intrinsics.h"

vrs4_t
__px_powi1_4(vrs4_t x, int32_t iy)
{
  return(__ZGVxN4v__mth_i_vr4si4(x, iy, __pmth_i_rpowi));
}

vrs4_t
__px_powi1_4m(vrs4_t x, int32_t iy, vis4_t mask)
{
  return(__ZGVxM4v__mth_i_vr4si4(x, iy, mask, __pmth_i_rpowi));
}

vrs4_t
__px_powi_4(vrs4_t x, vis4_t iy)
{
  return(__ZGVxN4vv__mth_i_vr4vi4(x, iy, __pmth_i_rpowi));
}

vrs4_t
__px_powi_4m(vrs4_t x, vis4_t iy, vis4_t mask)
{
  return(__ZGVxM4vv__mth_i_vr4vi4(x, iy, mask, __pmth_i_rpowi));
}

vrs4_t
__px_powk1_4(vrs4_t x, long long iy)
{
  return(__ZGVxN4v__mth_i_vr4si8(x, iy, __pmth_i_rpowk));
}

vrs4_t
__px_powk1_4m(vrs4_t x, long long iy, vis4_t mask)
{
  return(__ZGVxM4v__mth_i_vr4si8(x, iy, mask, __pmth_i_rpowk));
}

vrs4_t
__px_powk_4(vrs4_t x, vid2_t iyu, vid2_t iyl)
{
  return(__ZGVxN4vv__mth_i_vr4vi8(x, iyu, iyl, __pmth_i_rpowk));
}

vrs4_t
__px_powk_4m(vrs4_t x, vid2_t iyu, vid2_t iyl, vis4_t mask)
{
  return(__ZGVxM4vv__mth_i_vr4vi8(x, iyu, iyl, mask, __pmth_i_rpowk));
}

vrd2_t
__px_powi1_2(vrd2_t x, int32_t iy)
{
  return(__ZGVxN2v__mth_i_vr8si4(x, iy, __pmth_i_dpowi));
}

vrd2_t
__px_powi1_2m(vrd2_t x, int32_t iy, vid2_t mask)
{
  return(__ZGVxM2v__mth_i_vr8si4(x, iy, mask, __pmth_i_dpowi));
}

/*
 * __px_powi_2 and __px_powi_2m should technically be defined as:
 * __px_powi_2(vrd2_t x, vis2_t iy)
 * __px_powi_2m(vrd2_t x, vis2_t iy, vid2_t mask)
 *
 * But the POWER architectures needs the 32-bit integer vectors to
 * be the full 128-bits of a vector register.
 */

vrd2_t
__px_powi_2(vrd2_t x, vis4_t iy)
{
  return(__ZGVxN2vv__mth_i_vr8vi4(x, iy, __pmth_i_dpowi));
}

vrd2_t
__px_powi_2m(vrd2_t x, vis4_t iy, vid2_t mask)
{
  return(__ZGVxM2vv__mth_i_vr8vi4(x, iy, mask, __pmth_i_dpowi));
}

vrd2_t
__px_powk1_2(vrd2_t x, long long iy)
{
  return(__ZGVxN2v__mth_i_vr8si8(x, iy, __pmth_i_dpowk));
}

vrd2_t
__px_powk1_2m(vrd2_t x, long long iy, vid2_t mask)
{
  return(__ZGVxM2v__mth_i_vr8si8(x, iy, mask, __pmth_i_dpowk));
}

vrd2_t
__px_powk_2(vrd2_t x, vid2_t iy)
{
  return(__ZGVxN2vv__mth_i_vr8vi8(x, iy, __pmth_i_dpowk));
}

vrd2_t
__px_powk_2m(vrd2_t x, vid2_t iy, vid2_t mask)
{
  return(__ZGVxM2vv__mth_i_vr8vi8(x, iy, mask, __pmth_i_dpowk));
}
