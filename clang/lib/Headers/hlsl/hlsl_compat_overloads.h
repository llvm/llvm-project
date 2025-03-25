//===--- hlsl_compat_overloads.h - Extra HLSL overloads for intrinsics ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_COMPAT_OVERLOADS_H_
#define _HLSl_COMPAT_OVERLOADS_H_

namespace hlsl {

// Note: Functions in this file are sorted alphabetically, then grouped by base
// element type, and the element types are sorted by size, then singed integer,
// unsigned integer and floating point. Keeping this ordering consistent will
// help keep this file manageable as it grows.

#define _DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(fn)                                 \
  constexpr float fn(double V) { return fn((float)V); }                        \
  constexpr float2 fn(double2 V) { return fn((float2)V); }                     \
  constexpr float3 fn(double3 V) { return fn((float3)V); }                     \
  constexpr float4 fn(double4 V) { return fn((float4)V); }

#define _DXC_COMPAT_BINARY_DOUBLE_OVERLOADS(fn)                                \
  constexpr float fn(double V1, double V2) {                                   \
    return fn((float)V1, (float)V2);                                           \
  }                                                                            \
  constexpr float2 fn(double2 V1, double2 V2) {                                \
    return fn((float2)V1, (float2)V2);                                         \
  }                                                                            \
  constexpr float3 fn(double3 V1, double3 V2) {                                \
    return fn((float3)V1, (float3)V2);                                         \
  }                                                                            \
  constexpr float4 fn(double4 V1, double4 V2) {                                \
    return fn((float4)V1, (float4)V2);                                         \
  }

#define _DXC_COMPAT_TERNARY_DOUBLE_OVERLOADS(fn)                               \
  constexpr float fn(double V1, double V2, double V3) {                        \
    return fn((float)V1, (float)V2, (float)V3);                                \
  }                                                                            \
  constexpr float2 fn(double2 V1, double2 V2, double2 V3) {                    \
    return fn((float2)V1, (float2)V2, (float2)V3);                             \
  }                                                                            \
  constexpr float3 fn(double3 V1, double3 V2, double3 V3) {                    \
    return fn((float3)V1, (float3)V2, (float3)V3);                             \
  }                                                                            \
  constexpr float4 fn(double4 V1, double4 V2, double4 V3) {                    \
    return fn((float4)V1, (float4)V2, (float4)V3);                             \
  }

//===----------------------------------------------------------------------===//
// acos builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(acos)

//===----------------------------------------------------------------------===//
// asin builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(asin)

//===----------------------------------------------------------------------===//
// atan builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(atan)

//===----------------------------------------------------------------------===//
// atan2 builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_BINARY_DOUBLE_OVERLOADS(atan2)

//===----------------------------------------------------------------------===//
// ceil builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(ceil)

//===----------------------------------------------------------------------===//
// clamp builtins overloads
//===----------------------------------------------------------------------===//

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
clamp(vector<T, N> p0, vector<T, N> p1, T p2) {
  return clamp(p0, p1, (vector<T, N>)p2);
}

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
clamp(vector<T, N> p0, T p1, vector<T, N> p2) {
  return clamp(p0, (vector<T, N>)p1, p2);
}

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
clamp(vector<T, N> p0, T p1, T p2) {
  return clamp(p0, (vector<T, N>)p1, (vector<T, N>)p2);
}

//===----------------------------------------------------------------------===//
// cos builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(cos)

//===----------------------------------------------------------------------===//
// cosh builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(cosh)

//===----------------------------------------------------------------------===//
// degrees builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(degrees)

//===----------------------------------------------------------------------===//
// exp builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(exp)

//===----------------------------------------------------------------------===//
// exp2 builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(exp2)

//===----------------------------------------------------------------------===//
// floor builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(floor)

//===----------------------------------------------------------------------===//
// frac builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(frac)

//===----------------------------------------------------------------------===//
// isinf builtins overloads
//===----------------------------------------------------------------------===//

constexpr bool isinf(double V) { return isinf((float)V); }
constexpr bool2 isinf(double2 V) { return isinf((float2)V); }
constexpr bool3 isinf(double3 V) { return isinf((float3)V); }
constexpr bool4 isinf(double4 V) { return isinf((float4)V); }

//===----------------------------------------------------------------------===//
// lerp builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_TERNARY_DOUBLE_OVERLOADS(lerp)

//===----------------------------------------------------------------------===//
// log builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(log)

//===----------------------------------------------------------------------===//
// log10 builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(log10)

//===----------------------------------------------------------------------===//
// log2 builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(log2)

//===----------------------------------------------------------------------===//
// max builtins overloads
//===----------------------------------------------------------------------===//

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
max(vector<T, N> p0, T p1) {
  return max(p0, (vector<T, N>)p1);
}

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
max(T p0, vector<T, N> p1) {
  return max((vector<T, N>)p0, p1);
}

//===----------------------------------------------------------------------===//
// min builtins overloads
//===----------------------------------------------------------------------===//

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
min(vector<T, N> p0, T p1) {
  return min(p0, (vector<T, N>)p1);
}

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
min(T p0, vector<T, N> p1) {
  return min((vector<T, N>)p0, p1);
}

//===----------------------------------------------------------------------===//
// normalize builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(normalize)

//===----------------------------------------------------------------------===//
// pow builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_BINARY_DOUBLE_OVERLOADS(pow)

//===----------------------------------------------------------------------===//
// rsqrt builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(rsqrt)

//===----------------------------------------------------------------------===//
// round builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(round)

//===----------------------------------------------------------------------===//
// sin builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(sin)

//===----------------------------------------------------------------------===//
// sinh builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(sinh)

//===----------------------------------------------------------------------===//
// sqrt builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(sqrt)

//===----------------------------------------------------------------------===//
// step builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_BINARY_DOUBLE_OVERLOADS(step)

//===----------------------------------------------------------------------===//
// tan builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(tan)

//===----------------------------------------------------------------------===//
// tanh builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(tanh)

//===----------------------------------------------------------------------===//
// trunc builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(trunc)

//===----------------------------------------------------------------------===//
// radians builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(radians)

} // namespace hlsl
#endif // _HLSL_COMPAT_OVERLOADS_H_
