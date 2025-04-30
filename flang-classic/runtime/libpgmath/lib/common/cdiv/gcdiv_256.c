/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "mth_intrinsics.h"

vcs4_t
__gc_div_4(vcs4_t x, vcs4_t y)
{
  return (__ZGVyN4vv__mth_i_vc4vc4(x, y, __mth_i_cdiv_c99));
}

vcd2_t
__gz_div_2(vcd2_t x, vcd2_t y)
{
  return (__ZGVyN2vv__mth_i_vc8vc8(x, y, __mth_i_cddiv_c99));
}
