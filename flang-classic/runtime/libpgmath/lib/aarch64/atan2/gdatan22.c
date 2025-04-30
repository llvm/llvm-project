/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mth_intrinsics.h"

vrd2_t
__gd_atan2_2(vrd2_t x, vrd2_t y)
{
  return (__ZGVxN2vv__mth_i_vr8vr8(x, y, __mth_i_datan2));
}

vrd2_t
__gd_atan2_2m(vrd2_t x, vrd2_t y, vid2_t mask)
{
  return (__ZGVxM2vv__mth_i_vr8vr8(x, y, mask, __mth_i_datan2));
}
