/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "mth_intrinsics.h"

vcs8_t
__gc_sqrt_8(vcs8_t x)
{
  return (__ZGVzN8v__mth_i_vc4(x, csqrtf));
}

vcd4_t
__gz_sqrt_4(vcd4_t x)
{
  return (__ZGVzN4v__mth_i_vc8(x, csqrt));
}
