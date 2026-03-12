//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/math/clc_div_fast.h"
#include "clc/math/clc_ep.h"
#include "clc/math/clc_fma.h"
#include "clc/math/clc_ldexp.h"
#include "clc/math/clc_recip_fast.h"
#include "clc/math/clc_sqrt_fast.h"
#include "clc/relational/clc_isinf.h"
#include "clc/relational/clc_signbit.h"

#ifdef cl_khr_fp16
_CLC_DEF _CLC_OVERLOAD _CLC_CONST static half ep_high_fp_bits(half x) {
  return __clc_as_half((ushort)(__clc_as_ushort(x) & (ushort)0xffc0U));
}
#endif

_CLC_DEF _CLC_OVERLOAD _CLC_CONST static float ep_high_fp_bits(float x) {
  return __clc_as_float(__clc_as_uint(x) & 0xfffff000U);
}

#ifdef cl_khr_fp64

_CLC_DEF _CLC_OVERLOAD _CLC_CONST static double ep_high_fp_bits(double x) {
  return __clc_as_double(__clc_as_ulong(x) & 0xfffffffff8000000UL);
}
#endif

#define __CLC_BODY <clc_ep.inc>
#include <clc/math/gentype.inc>
