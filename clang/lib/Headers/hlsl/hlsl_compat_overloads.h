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
  constexpr float fn(double V) { return fn((float)V); }    \
  constexpr float2 fn(double2 V) { return fn((float2)V); } \
  constexpr float3 fn(double3 V) { return fn((float3)V); } \
  constexpr float4 fn(double4 V) { return fn((float4)V); }

#define _DXC_COMPAT_BINARY_DOUBLE_OVERLOADS(fn)                                \
  constexpr float fn(double V1, double V2) {                                   \
    return fn((float)V1, (float)V2);   \
  }                                                                            \
  constexpr float2 fn(double2 V1, double2 V2) {                                \
    return fn((float2)V1, (float2)V2); \
  }                                                                            \
  constexpr float3 fn(double3 V1, double3 V2) {                                \
    return fn((float3)V1, (float3)V2); \
  }                                                                            \
  constexpr float4 fn(double4 V1, double4 V2) {                                \
    return fn((float4)V1, (float4)V2); \
  }

#define _DXC_COMPAT_TERNARY_DOUBLE_OVERLOADS(fn)                               \
  constexpr float fn(double V1, double V2, double V3) {                        \
    return fn((float)V1, (float)V2,    \
              (float)V3);                                  \
  }                                                                            \
  constexpr float2 fn(double2 V1, double2 V2, double2 V3) {                    \
    return fn((float2)V1, (float2)V2,  \
              (float2)V3);                                 \
  }                                                                            \
  constexpr float3 fn(double3 V1, double3 V2, double3 V3) {                    \
    return fn((float3)V1, (float3)V2,  \
              (float3)V3);                                 \
  }                                                                            \
  constexpr float4 fn(double4 V1, double4 V2, double4 V3) {                    \
    return fn((float4)V1, (float4)V2,  \
              (float4)V3);                                 \
  }

//===----------------------------------------------------------------------===//
// acos builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(acos)
#endif

//===----------------------------------------------------------------------===//
// asin builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(asin)
#endif

//===----------------------------------------------------------------------===//
// atan builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(atan)
#endif

//===----------------------------------------------------------------------===//
// atan2 builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_BINARY_DOUBLE_OVERLOADS(atan2)
#endif

//===----------------------------------------------------------------------===//
// ceil builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(ceil)
#endif

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

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(cos)
#endif

//===----------------------------------------------------------------------===//
// cosh builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(cosh)
#endif

//===----------------------------------------------------------------------===//
// degrees builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(degrees)
#endif

//===----------------------------------------------------------------------===//
// exp builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(exp)
#endif

//===----------------------------------------------------------------------===//
// exp2 builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(exp2)
#endif

//===----------------------------------------------------------------------===//
// floor builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(floor)
#endif

//===----------------------------------------------------------------------===//
// frac builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(frac)
#endif

//===----------------------------------------------------------------------===//
// isinf builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
constexpr bool isinf(double V) { return isinf((float)V); }
constexpr bool2 isinf(double2 V) {
  return isinf((float2)V);
}
constexpr bool3 isinf(double3 V) {
  return isinf((float3)V);
}
constexpr bool4 isinf(double4 V) {
  return isinf((float4)V);
}
#endif

//===----------------------------------------------------------------------===//
// lerp builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_TERNARY_DOUBLE_OVERLOADS(lerp)
#endif

//===----------------------------------------------------------------------===//
// log builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(log)
#endif

//===----------------------------------------------------------------------===//
// log10 builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(log10)
#endif

//===----------------------------------------------------------------------===//
// log2 builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(log2)
#endif

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

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(normalize)
#endif

//===----------------------------------------------------------------------===//
// pow builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_BINARY_DOUBLE_OVERLOADS(pow)
#endif

//===----------------------------------------------------------------------===//
// rsqrt builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(rsqrt)
#endif

//===----------------------------------------------------------------------===//
// round builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(round)
#endif

//===----------------------------------------------------------------------===//
// sin builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(sin)
#endif

//===----------------------------------------------------------------------===//
// sinh builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(sinh)
#endif

//===----------------------------------------------------------------------===//
// sqrt builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(sqrt)
#endif

//===----------------------------------------------------------------------===//
// step builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_BINARY_DOUBLE_OVERLOADS(step)
#endif

//===----------------------------------------------------------------------===//
// tan builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(tan)
#endif

//===----------------------------------------------------------------------===//
// tanh builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(tanh)
#endif

//===----------------------------------------------------------------------===//
// trunc builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(trunc)
#endif

//===----------------------------------------------------------------------===//
// radians builtins overloads
//===----------------------------------------------------------------------===//

#if __HLSL_VERSION <= __HLSL_202x
_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(radians)
#endif

} // namespace hlsl
#endif // _HLSL_COMPAT_OVERLOADS_H_
