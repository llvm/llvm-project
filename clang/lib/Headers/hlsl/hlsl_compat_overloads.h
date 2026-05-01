//===--- hlsl_compat_overloads.h - Extra HLSL overloads for intrinsics ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_COMPAT_OVERLOADS_H_
#define _HLSL_COMPAT_OVERLOADS_H_

namespace hlsl {

// Note: Functions in this file are sorted alphabetically, then grouped by base
// element type, and the element types are sorted by size, then signed integer,
// unsigned integer and floating point. Keeping this ordering consistent will
// help keep this file manageable as it grows.

#define _DXC_DEPRECATED_64BIT_FN(fn)                                           \
  [[deprecated("In 202x 64 bit API lowering for " #fn " is deprecated. "       \
               "Explicitly cast parameters to 32 or 16 bit types.")]]

#define _DXC_DEPRECATED_INT_FN(fn)                                             \
  [[deprecated("In 202x int lowering for " #fn " is deprecated. "              \
               "Explicitly cast parameters to float types.")]]

#define _DXC_DEPRECATED_VEC_SCALAR_FN(fn)                                      \
  [[deprecated("In 202x mismatched vector/scalar lowering for " #fn " is "     \
               "deprecated. Explicitly cast parameters.")]]

#define _DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(fn)                                 \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float fn(double V) { return fn((float)V); }                        \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float2 fn(double2 V) { return fn((float2)V); }                     \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float3 fn(double3 V) { return fn((float3)V); }                     \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float4 fn(double4 V) { return fn((float4)V); }

#define _DXC_COMPAT_BINARY_DOUBLE_OVERLOADS(fn)                                \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float fn(double V1, double V2) {                                   \
    return fn((float)V1, (float)V2);                                           \
  }                                                                            \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float2 fn(double2 V1, double2 V2) {                                \
    return fn((float2)V1, (float2)V2);                                         \
  }                                                                            \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float3 fn(double3 V1, double3 V2) {                                \
    return fn((float3)V1, (float3)V2);                                         \
  }                                                                            \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float4 fn(double4 V1, double4 V2) {                                \
    return fn((float4)V1, (float4)V2);                                         \
  }

#define _DXC_COMPAT_TERNARY_DOUBLE_OVERLOADS(fn)                               \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float fn(double V1, double V2, double V3) {                        \
    return fn((float)V1, (float)V2, (float)V3);                                \
  }                                                                            \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float2 fn(double2 V1, double2 V2, double2 V3) {                    \
    return fn((float2)V1, (float2)V2, (float2)V3);                             \
  }                                                                            \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float3 fn(double3 V1, double3 V2, double3 V3) {                    \
    return fn((float3)V1, (float3)V2, (float3)V3);                             \
  }                                                                            \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float4 fn(double4 V1, double4 V2, double4 V3) {                    \
    return fn((float4)V1, (float4)V2, (float4)V3);                             \
  }

#define _DXC_COMPAT_UNARY_INTEGER_OVERLOADS(fn)                                \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float fn(int V) { return fn((float)V); }                           \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2 fn(int2 V) { return fn((float2)V); }                        \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3 fn(int3 V) { return fn((float3)V); }                        \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4 fn(int4 V) { return fn((float4)V); }                        \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float fn(uint V) { return fn((float)V); }                          \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2 fn(uint2 V) { return fn((float2)V); }                       \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3 fn(uint3 V) { return fn((float3)V); }                       \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4 fn(uint4 V) { return fn((float4)V); }                       \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float fn(int64_t V) { return fn((float)V); }                       \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2 fn(int64_t2 V) { return fn((float2)V); }                    \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3 fn(int64_t3 V) { return fn((float3)V); }                    \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4 fn(int64_t4 V) { return fn((float4)V); }                    \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float fn(uint64_t V) { return fn((float)V); }                      \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2 fn(uint64_t2 V) { return fn((float2)V); }                   \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3 fn(uint64_t3 V) { return fn((float3)V); }                   \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4 fn(uint64_t4 V) { return fn((float4)V); }

#define _DXC_COMPAT_BINARY_INTEGER_OVERLOADS(fn)                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float fn(int V1, int V2) { return fn((float)V1, (float)V2); }      \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2 fn(int2 V1, int2 V2) { return fn((float2)V1, (float2)V2); } \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3 fn(int3 V1, int3 V2) { return fn((float3)V1, (float3)V2); } \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4 fn(int4 V1, int4 V2) { return fn((float4)V1, (float4)V2); } \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float fn(uint V1, uint V2) { return fn((float)V1, (float)V2); }    \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2 fn(uint2 V1, uint2 V2) {                                    \
    return fn((float2)V1, (float2)V2);                                         \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3 fn(uint3 V1, uint3 V2) {                                    \
    return fn((float3)V1, (float3)V2);                                         \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4 fn(uint4 V1, uint4 V2) {                                    \
    return fn((float4)V1, (float4)V2);                                         \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float fn(int64_t V1, int64_t V2) {                                 \
    return fn((float)V1, (float)V2);                                           \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2 fn(int64_t2 V1, int64_t2 V2) {                              \
    return fn((float2)V1, (float2)V2);                                         \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3 fn(int64_t3 V1, int64_t3 V2) {                              \
    return fn((float3)V1, (float3)V2);                                         \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4 fn(int64_t4 V1, int64_t4 V2) {                              \
    return fn((float4)V1, (float4)V2);                                         \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float fn(uint64_t V1, uint64_t V2) {                               \
    return fn((float)V1, (float)V2);                                           \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2 fn(uint64_t2 V1, uint64_t2 V2) {                            \
    return fn((float2)V1, (float2)V2);                                         \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3 fn(uint64_t3 V1, uint64_t3 V2) {                            \
    return fn((float3)V1, (float3)V2);                                         \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4 fn(uint64_t4 V1, uint64_t4 V2) {                            \
    return fn((float4)V1, (float4)V2);                                         \
  }

#define _DXC_COMPAT_TERNARY_INTEGER_OVERLOADS(fn)                              \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float fn(int V1, int V2, int V3) {                                 \
    return fn((float)V1, (float)V2, (float)V3);                                \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2 fn(int2 V1, int2 V2, int2 V3) {                             \
    return fn((float2)V1, (float2)V2, (float2)V3);                             \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3 fn(int3 V1, int3 V2, int3 V3) {                             \
    return fn((float3)V1, (float3)V2, (float3)V3);                             \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4 fn(int4 V1, int4 V2, int4 V3) {                             \
    return fn((float4)V1, (float4)V2, (float4)V3);                             \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float fn(uint V1, uint V2, uint V3) {                              \
    return fn((float)V1, (float)V2, (float)V3);                                \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2 fn(uint2 V1, uint2 V2, uint2 V3) {                          \
    return fn((float2)V1, (float2)V2, (float2)V3);                             \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3 fn(uint3 V1, uint3 V2, uint3 V3) {                          \
    return fn((float3)V1, (float3)V2, (float3)V3);                             \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4 fn(uint4 V1, uint4 V2, uint4 V3) {                          \
    return fn((float4)V1, (float4)V2, (float4)V3);                             \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float fn(int64_t V1, int64_t V2, int64_t V3) {                     \
    return fn((float)V1, (float)V2, (float)V3);                                \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2 fn(int64_t2 V1, int64_t2 V2, int64_t2 V3) {                 \
    return fn((float2)V1, (float2)V2, (float2)V3);                             \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3 fn(int64_t3 V1, int64_t3 V2, int64_t3 V3) {                 \
    return fn((float3)V1, (float3)V2, (float3)V3);                             \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4 fn(int64_t4 V1, int64_t4 V2, int64_t4 V3) {                 \
    return fn((float4)V1, (float4)V2, (float4)V3);                             \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float fn(uint64_t V1, uint64_t V2, uint64_t V3) {                  \
    return fn((float)V1, (float)V2, (float)V3);                                \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2 fn(uint64_t2 V1, uint64_t2 V2, uint64_t2 V3) {              \
    return fn((float2)V1, (float2)V2, (float2)V3);                             \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3 fn(uint64_t3 V1, uint64_t3 V2, uint64_t3 V3) {              \
    return fn((float3)V1, (float3)V2, (float3)V3);                             \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4 fn(uint64_t4 V1, uint64_t4 V2, uint64_t4 V3) {              \
    return fn((float4)V1, (float4)V2, (float4)V3);                             \
  }

#define _DXC_COMPAT_BINARY_DOUBLE_MATRIX_OVERLOADS(fn)                         \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float1x1 fn(double1x1 y, double1x1 x) {                            \
    return fn((float1x1)y, (float1x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float1x2 fn(double1x2 y, double1x2 x) {                            \
    return fn((float1x2)y, (float1x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float1x3 fn(double1x3 y, double1x3 x) {                            \
    return fn((float1x3)y, (float1x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float1x4 fn(double1x4 y, double1x4 x) {                            \
    return fn((float1x4)y, (float1x4)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float2x1 fn(double2x1 y, double2x1 x) {                            \
    return fn((float2x1)y, (float2x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float2x2 fn(double2x2 y, double2x2 x) {                            \
    return fn((float2x2)y, (float2x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float2x3 fn(double2x3 y, double2x3 x) {                            \
    return fn((float2x3)y, (float2x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float2x4 fn(double2x4 y, double2x4 x) {                            \
    return fn((float2x4)y, (float2x4)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float3x1 fn(double3x1 y, double3x1 x) {                            \
    return fn((float3x1)y, (float3x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float3x2 fn(double3x2 y, double3x2 x) {                            \
    return fn((float3x2)y, (float3x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float3x3 fn(double3x3 y, double3x3 x) {                            \
    return fn((float3x3)y, (float3x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float3x4 fn(double3x4 y, double3x4 x) {                            \
    return fn((float3x4)y, (float3x4)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float4x1 fn(double4x1 y, double4x1 x) {                            \
    return fn((float4x1)y, (float4x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float4x2 fn(double4x2 y, double4x2 x) {                            \
    return fn((float4x2)y, (float4x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float4x3 fn(double4x3 y, double4x3 x) {                            \
    return fn((float4x3)y, (float4x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_64BIT_FN(fn)                                                 \
  constexpr float4x4 fn(double4x4 y, double4x4 x) {                            \
    return fn((float4x4)y, (float4x4)x);                                       \
  }

#define _DXC_COMPAT_BINARY_INTEGER_MATRIX_OVERLOADS(fn)                        \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float1x1 fn(int1x1 y, int1x1 x) {                                  \
    return fn((float1x1)y, (float1x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float1x2 fn(int1x2 y, int1x2 x) {                                  \
    return fn((float1x2)y, (float1x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float1x3 fn(int1x3 y, int1x3 x) {                                  \
    return fn((float1x3)y, (float1x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float1x4 fn(int1x4 y, int1x4 x) {                                  \
    return fn((float1x4)y, (float1x4)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2x1 fn(int2x1 y, int2x1 x) {                                  \
    return fn((float2x1)y, (float2x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2x2 fn(int2x2 y, int2x2 x) {                                  \
    return fn((float2x2)y, (float2x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2x3 fn(int2x3 y, int2x3 x) {                                  \
    return fn((float2x3)y, (float2x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2x4 fn(int2x4 y, int2x4 x) {                                  \
    return fn((float2x4)y, (float2x4)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3x1 fn(int3x1 y, int3x1 x) {                                  \
    return fn((float3x1)y, (float3x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3x2 fn(int3x2 y, int3x2 x) {                                  \
    return fn((float3x2)y, (float3x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3x3 fn(int3x3 y, int3x3 x) {                                  \
    return fn((float3x3)y, (float3x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3x4 fn(int3x4 y, int3x4 x) {                                  \
    return fn((float3x4)y, (float3x4)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4x1 fn(int4x1 y, int4x1 x) {                                  \
    return fn((float4x1)y, (float4x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4x2 fn(int4x2 y, int4x2 x) {                                  \
    return fn((float4x2)y, (float4x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4x3 fn(int4x3 y, int4x3 x) {                                  \
    return fn((float4x3)y, (float4x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4x4 fn(int4x4 y, int4x4 x) {                                  \
    return fn((float4x4)y, (float4x4)x);                                       \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float1x1 fn(uint1x1 y, uint1x1 x) {                                \
    return fn((float1x1)y, (float1x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float1x2 fn(uint1x2 y, uint1x2 x) {                                \
    return fn((float1x2)y, (float1x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float1x3 fn(uint1x3 y, uint1x3 x) {                                \
    return fn((float1x3)y, (float1x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float1x4 fn(uint1x4 y, uint1x4 x) {                                \
    return fn((float1x4)y, (float1x4)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2x1 fn(uint2x1 y, uint2x1 x) {                                \
    return fn((float2x1)y, (float2x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2x2 fn(uint2x2 y, uint2x2 x) {                                \
    return fn((float2x2)y, (float2x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2x3 fn(uint2x3 y, uint2x3 x) {                                \
    return fn((float2x3)y, (float2x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2x4 fn(uint2x4 y, uint2x4 x) {                                \
    return fn((float2x4)y, (float2x4)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3x1 fn(uint3x1 y, uint3x1 x) {                                \
    return fn((float3x1)y, (float3x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3x2 fn(uint3x2 y, uint3x2 x) {                                \
    return fn((float3x2)y, (float3x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3x3 fn(uint3x3 y, uint3x3 x) {                                \
    return fn((float3x3)y, (float3x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3x4 fn(uint3x4 y, uint3x4 x) {                                \
    return fn((float3x4)y, (float3x4)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4x1 fn(uint4x1 y, uint4x1 x) {                                \
    return fn((float4x1)y, (float4x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4x2 fn(uint4x2 y, uint4x2 x) {                                \
    return fn((float4x2)y, (float4x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4x3 fn(uint4x3 y, uint4x3 x) {                                \
    return fn((float4x3)y, (float4x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4x4 fn(uint4x4 y, uint4x4 x) {                                \
    return fn((float4x4)y, (float4x4)x);                                       \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float1x1 fn(int64_t1x1 y, int64_t1x1 x) {                          \
    return fn((float1x1)y, (float1x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float1x2 fn(int64_t1x2 y, int64_t1x2 x) {                          \
    return fn((float1x2)y, (float1x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float1x3 fn(int64_t1x3 y, int64_t1x3 x) {                          \
    return fn((float1x3)y, (float1x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float1x4 fn(int64_t1x4 y, int64_t1x4 x) {                          \
    return fn((float1x4)y, (float1x4)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2x1 fn(int64_t2x1 y, int64_t2x1 x) {                          \
    return fn((float2x1)y, (float2x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2x2 fn(int64_t2x2 y, int64_t2x2 x) {                          \
    return fn((float2x2)y, (float2x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2x3 fn(int64_t2x3 y, int64_t2x3 x) {                          \
    return fn((float2x3)y, (float2x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2x4 fn(int64_t2x4 y, int64_t2x4 x) {                          \
    return fn((float2x4)y, (float2x4)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3x1 fn(int64_t3x1 y, int64_t3x1 x) {                          \
    return fn((float3x1)y, (float3x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3x2 fn(int64_t3x2 y, int64_t3x2 x) {                          \
    return fn((float3x2)y, (float3x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3x3 fn(int64_t3x3 y, int64_t3x3 x) {                          \
    return fn((float3x3)y, (float3x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3x4 fn(int64_t3x4 y, int64_t3x4 x) {                          \
    return fn((float3x4)y, (float3x4)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4x1 fn(int64_t4x1 y, int64_t4x1 x) {                          \
    return fn((float4x1)y, (float4x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4x2 fn(int64_t4x2 y, int64_t4x2 x) {                          \
    return fn((float4x2)y, (float4x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4x3 fn(int64_t4x3 y, int64_t4x3 x) {                          \
    return fn((float4x3)y, (float4x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4x4 fn(int64_t4x4 y, int64_t4x4 x) {                          \
    return fn((float4x4)y, (float4x4)x);                                       \
  }                                                                            \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float1x1 fn(uint64_t1x1 y, uint64_t1x1 x) {                        \
    return fn((float1x1)y, (float1x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float1x2 fn(uint64_t1x2 y, uint64_t1x2 x) {                        \
    return fn((float1x2)y, (float1x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float1x3 fn(uint64_t1x3 y, uint64_t1x3 x) {                        \
    return fn((float1x3)y, (float1x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float1x4 fn(uint64_t1x4 y, uint64_t1x4 x) {                        \
    return fn((float1x4)y, (float1x4)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2x1 fn(uint64_t2x1 y, uint64_t2x1 x) {                        \
    return fn((float2x1)y, (float2x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2x2 fn(uint64_t2x2 y, uint64_t2x2 x) {                        \
    return fn((float2x2)y, (float2x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2x3 fn(uint64_t2x3 y, uint64_t2x3 x) {                        \
    return fn((float2x3)y, (float2x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float2x4 fn(uint64_t2x4 y, uint64_t2x4 x) {                        \
    return fn((float2x4)y, (float2x4)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3x1 fn(uint64_t3x1 y, uint64_t3x1 x) {                        \
    return fn((float3x1)y, (float3x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3x2 fn(uint64_t3x2 y, uint64_t3x2 x) {                        \
    return fn((float3x2)y, (float3x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3x3 fn(uint64_t3x3 y, uint64_t3x3 x) {                        \
    return fn((float3x3)y, (float3x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float3x4 fn(uint64_t3x4 y, uint64_t3x4 x) {                        \
    return fn((float3x4)y, (float3x4)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4x1 fn(uint64_t4x1 y, uint64_t4x1 x) {                        \
    return fn((float4x1)y, (float4x1)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4x2 fn(uint64_t4x2 y, uint64_t4x2 x) {                        \
    return fn((float4x2)y, (float4x2)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4x3 fn(uint64_t4x3 y, uint64_t4x3 x) {                        \
    return fn((float4x3)y, (float4x3)x);                                       \
  }                                                                            \
                                                                               \
  _DXC_DEPRECATED_INT_FN(fn)                                                   \
  constexpr float4x4 fn(uint64_t4x4 y, uint64_t4x4 x) {                        \
    return fn((float4x4)y, (float4x4)x);                                       \
  }
//===----------------------------------------------------------------------===//
// acos builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(acos)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(acos)

//===----------------------------------------------------------------------===//
// asin builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(asin)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(asin)

//===----------------------------------------------------------------------===//
// atan builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(atan)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(atan)

//===----------------------------------------------------------------------===//
// atan2 builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_BINARY_DOUBLE_OVERLOADS(atan2)
_DXC_COMPAT_BINARY_INTEGER_OVERLOADS(atan2)
_DXC_COMPAT_BINARY_DOUBLE_MATRIX_OVERLOADS(atan2)
_DXC_COMPAT_BINARY_INTEGER_MATRIX_OVERLOADS(atan2)

//===----------------------------------------------------------------------===//
// ceil builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(ceil)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(ceil)

//===----------------------------------------------------------------------===//
// clamp builtins overloads
//===----------------------------------------------------------------------===//

template <typename T, uint N>
_DXC_DEPRECATED_VEC_SCALAR_FN(clamp)
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>> clamp(
    vector<T, N> p0, vector<T, N> p1, T p2) {
  return clamp(p0, p1, (vector<T, N>)p2);
}

template <typename T, uint N>
_DXC_DEPRECATED_VEC_SCALAR_FN(clamp)
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>> clamp(
    vector<T, N> p0, T p1, vector<T, N> p2) {
  return clamp(p0, (vector<T, N>)p1, p2);
}

template <typename T, uint N>
_DXC_DEPRECATED_VEC_SCALAR_FN(clamp)
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>> clamp(
    vector<T, N> p0, T p1, T p2) {
  return clamp(p0, (vector<T, N>)p1, (vector<T, N>)p2);
}

//===----------------------------------------------------------------------===//
// cos builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(cos)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(cos)

//===----------------------------------------------------------------------===//
// cosh builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(cosh)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(cosh)

//===----------------------------------------------------------------------===//
// degrees builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(degrees)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(degrees)

//===----------------------------------------------------------------------===//
// exp builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(exp)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(exp)

//===----------------------------------------------------------------------===//
// exp2 builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(exp2)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(exp2)

//===----------------------------------------------------------------------===//
// floor builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(floor)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(floor)

//===----------------------------------------------------------------------===//
// frac builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(frac)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(frac)

//===----------------------------------------------------------------------===//
// isinf builtins overloads
//===----------------------------------------------------------------------===//

_DXC_DEPRECATED_64BIT_FN(fn)
constexpr bool isinf(double V) { return isinf((float)V); }
_DXC_DEPRECATED_64BIT_FN(fn)
constexpr bool2 isinf(double2 V) { return isinf((float2)V); }
_DXC_DEPRECATED_64BIT_FN(fn)
constexpr bool3 isinf(double3 V) { return isinf((float3)V); }
_DXC_DEPRECATED_64BIT_FN(fn)
constexpr bool4 isinf(double4 V) { return isinf((float4)V); }

//===----------------------------------------------------------------------===//
// isnan builtins overloads
//===----------------------------------------------------------------------===//

constexpr bool isnan(double V) { return isnan((float)V); }
constexpr bool2 isnan(double2 V) { return isnan((float2)V); }
constexpr bool3 isnan(double3 V) { return isnan((float3)V); }
constexpr bool4 isnan(double4 V) { return isnan((float4)V); }

//===----------------------------------------------------------------------===//
// lerp builtins overloads
//===----------------------------------------------------------------------===//

template <typename T, uint N>
_DXC_DEPRECATED_VEC_SCALAR_FN(lerp)
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>> lerp(
    vector<T, N> x, vector<T, N> y, T s) {
  return lerp(x, y, (vector<T, N>)s);
}

_DXC_COMPAT_TERNARY_DOUBLE_OVERLOADS(lerp)
_DXC_COMPAT_TERNARY_INTEGER_OVERLOADS(lerp)

//===----------------------------------------------------------------------===//
// log builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(log)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(log)

//===----------------------------------------------------------------------===//
// log10 builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(log10)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(log10)

//===----------------------------------------------------------------------===//
// log2 builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(log2)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(log2)

//===----------------------------------------------------------------------===//
// max builtins overloads
//===----------------------------------------------------------------------===//

template <typename T, uint N>
_DXC_DEPRECATED_VEC_SCALAR_FN(max)
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>> max(
    vector<T, N> p0, T p1) {
  return max(p0, (vector<T, N>)p1);
}

template <typename T, uint N>
_DXC_DEPRECATED_VEC_SCALAR_FN(max)
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>> max(
    T p0, vector<T, N> p1) {
  return max((vector<T, N>)p0, p1);
}

//===----------------------------------------------------------------------===//
// min builtins overloads
//===----------------------------------------------------------------------===//

template <typename T, uint N>
_DXC_DEPRECATED_VEC_SCALAR_FN(min)
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>> min(
    vector<T, N> p0, T p1) {
  return min(p0, (vector<T, N>)p1);
}

template <typename T, uint N>
_DXC_DEPRECATED_VEC_SCALAR_FN(min)
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>> min(
    T p0, vector<T, N> p1) {
  return min((vector<T, N>)p0, p1);
}

//===----------------------------------------------------------------------===//
// normalize builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(normalize)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(normalize)

//===----------------------------------------------------------------------===//
// pow builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_BINARY_DOUBLE_OVERLOADS(pow)
_DXC_COMPAT_BINARY_INTEGER_OVERLOADS(pow)

//===----------------------------------------------------------------------===//
// rsqrt builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(rsqrt)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(rsqrt)

//===----------------------------------------------------------------------===//
// round builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(round)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(round)

//===----------------------------------------------------------------------===//
// sin builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(sin)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(sin)

//===----------------------------------------------------------------------===//
// sinh builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(sinh)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(sinh)

//===----------------------------------------------------------------------===//
// sqrt builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(sqrt)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(sqrt)

//===----------------------------------------------------------------------===//
// step builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_BINARY_DOUBLE_OVERLOADS(step)
_DXC_COMPAT_BINARY_INTEGER_OVERLOADS(step)

//===----------------------------------------------------------------------===//
// tan builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(tan)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(tan)

//===----------------------------------------------------------------------===//
// tanh builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(tanh)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(tanh)

//===----------------------------------------------------------------------===//
// trunc builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(trunc)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(trunc)

//===----------------------------------------------------------------------===//
// radians builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(radians)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(radians)

} // namespace hlsl
#endif // _HLSL_COMPAT_OVERLOADS_H_
