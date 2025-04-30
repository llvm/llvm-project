/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "mth_intrinsics.h"

vcs8_t
__gc_div_8(vcs8_t x, vcs8_t y)
{
  return (__ZGVzN8vv__mth_i_vc4vc4(x, y, __mth_i_cdiv_c99));
}

vcd4_t
__gz_div_4(vcd4_t x, vcd4_t y)
{
  return (__ZGVzN4vv__mth_i_vc8vc8(x, y, __mth_i_cddiv_c99));
}
