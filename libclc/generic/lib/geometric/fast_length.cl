//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>

_CLC_OVERLOAD _CLC_DEF float fast_length(float p) {
  return fabs(p);
}

_CLC_OVERLOAD _CLC_DEF float fast_length(float2 p) {
  return half_sqrt(dot(p, p));
}

_CLC_OVERLOAD _CLC_DEF float fast_length(float3 p) {
  return half_sqrt(dot(p, p));
}

_CLC_OVERLOAD _CLC_DEF float fast_length(float4 p) {
  return half_sqrt(dot(p, p));
}
