/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mth_intrinsics.h"

vrs4_t
__gs_atan2_4(vrs4_t x, vrs4_t y)
{
  return (__ZGVxN4vv__mth_i_vr4vr4(x, y, __mth_i_atan2));
}

vrs4_t
__gs_atan2_4m(vrs4_t x, vrs4_t y, vis4_t mask)
{
  return (__ZGVxM4vv__mth_i_vr4vr4(x, y, mask, __mth_i_atan2));
}
