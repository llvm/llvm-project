/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "mth_intrinsics.h"

vcs4_t
__gc_sqrt_4(vcs4_t x)
{
  return (__ZGVyN4v__mth_i_vc4(x, csqrtf));
}

vcd2_t
__gz_sqrt_2(vcd2_t x)
{
  return (__ZGVyN2v__mth_i_vc8(x, csqrt));
}
