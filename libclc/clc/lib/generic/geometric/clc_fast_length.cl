//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/geometric/clc_dot.h>
#include <clc/internal/clc.h>
#include <clc/math/clc_fabs.h>
#include <clc/math/clc_half_sqrt.h>

_CLC_OVERLOAD _CLC_DEF float __clc_fast_length(float p) {
  return __clc_fabs(p);
}

_CLC_OVERLOAD _CLC_DEF float __clc_fast_length(float2 p) {
  return __clc_half_sqrt(__clc_dot(p, p));
}

_CLC_OVERLOAD _CLC_DEF float __clc_fast_length(float3 p) {
  return __clc_half_sqrt(__clc_dot(p, p));
}

_CLC_OVERLOAD _CLC_DEF float __clc_fast_length(float4 p) {
  return __clc_half_sqrt(__clc_dot(p, p));
}
