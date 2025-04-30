
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mth_intrinsics.h"

vrs8_t
__py_powi1_8(vrs8_t x, int32_t iy)
{
  return(__ZGVyN8v__mth_i_vr4si4(x, iy, __pmth_i_rpowi));
}

vrs8_t
__py_powi1_8m(vrs8_t x, int32_t iy, vis8_t mask)
{
  return(__ZGVyM8v__mth_i_vr4si4(x, iy, mask, __pmth_i_rpowi));
}

vrs8_t
__py_powi_8(vrs8_t x, vis8_t iy)
{
  return(__ZGVyN8vv__mth_i_vr4vi4(x, iy, __pmth_i_rpowi));
}

vrs8_t
__py_powi_8m(vrs8_t x, vis8_t iy, vis8_t mask)
{
  return(__ZGVyM8vv__mth_i_vr4vi4(x, iy, mask, __pmth_i_rpowi));
}

vrs8_t
__py_powk1_8(vrs8_t x, long long iy)
{
  return(__ZGVyN8v__mth_i_vr4si8(x, iy, __pmth_i_rpowk));
}

vrs8_t
__py_powk1_8m(vrs8_t x, long long iy, vis8_t mask)
{
  return(__ZGVyM8v__mth_i_vr4si8(x, iy, mask, __pmth_i_rpowk));
}

vrs8_t
__py_powk_8(vrs8_t x, vid4_t iyu, vid4_t iyl)
{
  return(__ZGVyN8vv__mth_i_vr4vi8(x, iyu, iyl, __pmth_i_rpowk));
}

vrs8_t
__py_powk_8m(vrs8_t x, vid4_t iyu, vid4_t iyl, vis8_t mask)
{
  return(__ZGVyM8vv__mth_i_vr4vi8(x, iyu, iyl, mask, __pmth_i_rpowk));
}

vrd4_t
__py_powi1_4(vrd4_t x, int32_t iy)
{
  return(__ZGVyN4v__mth_i_vr8si4(x, iy, __pmth_i_dpowi));
}

vrd4_t
__py_powi1_4m(vrd4_t x, int32_t iy, vid4_t mask)
{
  return(__ZGVyM4v__mth_i_vr8si4(x, iy, mask, __pmth_i_dpowi));
}

vrd4_t
__py_powi_4(vrd4_t x, vis4_t iy)
{
  return(__ZGVyN4vv__mth_i_vr8vi4(x, iy, __pmth_i_dpowi));
}

vrd4_t
__py_powi_4m(vrd4_t x, vis4_t iy, vid4_t mask)
{
  return(__ZGVyM4vv__mth_i_vr8vi4(x, iy, mask, __pmth_i_dpowi));
}

vrd4_t
__py_powk1_4(vrd4_t x, long long iy)
{
  return(__ZGVyN4v__mth_i_vr8si8(x, iy, __pmth_i_dpowk));
}

vrd4_t
__py_powk1_4m(vrd4_t x, long long iy, vid4_t mask)
{
  return(__ZGVyM4v__mth_i_vr8si8(x, iy, mask, __pmth_i_dpowk));
}

vrd4_t
__py_powk_4(vrd4_t x, vid4_t iy)
{
  return(__ZGVyN4vv__mth_i_vr8vi8(x, iy, __pmth_i_dpowk));
}

vrd4_t
__py_powk_4m(vrd4_t x, vid4_t iy, vid4_t mask)
{
  return(__ZGVyM4vv__mth_i_vr8vi8(x, iy, mask, __pmth_i_dpowk));
}
