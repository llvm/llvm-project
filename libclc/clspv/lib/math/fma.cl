//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/internal/math/clc_sw_fma.h>

_CLC_DEFINE_TERNARY_BUILTIN(float, fma, __clc_sw_fma, float, float, float)

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half fma(half a, half b, half c) {
  return (half)mad((float)a, (float)b, (float)c);
}
_CLC_TERNARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, half, fma, half, half, half)

#endif
