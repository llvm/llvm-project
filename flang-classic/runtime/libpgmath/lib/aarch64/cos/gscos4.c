/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mth_intrinsics.h"

vrs4_t
__gs_cos_4(vrs4_t x)
{
  return (__ZGVxN4v__mth_i_vr4(x, __mth_i_cos));
}

vrs4_t
__gs_cos_4m(vrs4_t x, vis4_t mask)
{
  return (__ZGVxM4v__mth_i_vr4(x, mask, __mth_i_cos));
}

vcs1_t
__gc_cos_1(vcs1_t x)
{
  return (__ZGVxN1v__mth_i_vc4(x, ccosf));
}

vcs2_t
__gc_cos_2(vcs2_t x)
{
  return (__ZGVxN2v__mth_i_vc4(x, ccosf));
}
