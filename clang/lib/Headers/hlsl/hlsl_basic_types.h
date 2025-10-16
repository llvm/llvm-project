//===----- hlsl_basic_types.h - HLSL definitions for basic types ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_BASIC_TYPES_H_
#define _HLSL_HLSL_BASIC_TYPES_H_

namespace hlsl {
// built-in scalar data types:

/// \typedef template<typename Ty, int Size> using vector = Ty
/// __attribute__((ext_vector_type(Size)))
///
/// \tparam Ty The base type of the vector may be any builtin integral or
/// floating point type.
/// \tparam Size The size of the vector may be any value between 1 and 4.

#ifdef __HLSL_ENABLE_16_BIT
// 16-bit integer.
typedef unsigned short uint16_t;
typedef short int16_t;

// 16-bit floating point.
typedef half float16_t;
#endif

// 32-bit integer.
typedef int int32_t;

// unsigned 32-bit integer.
typedef unsigned int uint;
typedef unsigned int uint32_t;

// 32-bit floating point.
typedef float float32_t;

// 64-bit integer.
typedef unsigned long uint64_t;
typedef long int64_t;

// 64-bit floating point
typedef double float64_t;

// built-in vector data types:

#ifdef __HLSL_ENABLE_16_BIT
typedef vector<int16_t, 1> int16_t1;
typedef vector<int16_t, 2> int16_t2;
typedef vector<int16_t, 3> int16_t3;
typedef vector<int16_t, 4> int16_t4;
typedef vector<uint16_t, 1> uint16_t1;
typedef vector<uint16_t, 2> uint16_t2;
typedef vector<uint16_t, 3> uint16_t3;
typedef vector<uint16_t, 4> uint16_t4;
#endif
typedef vector<bool, 1> bool1;
typedef vector<bool, 2> bool2;
typedef vector<bool, 3> bool3;
typedef vector<bool, 4> bool4;
typedef vector<int, 1> int1;
typedef vector<int, 2> int2;
typedef vector<int, 3> int3;
typedef vector<int, 4> int4;
typedef vector<uint, 1> uint1;
typedef vector<uint, 2> uint2;
typedef vector<uint, 3> uint3;
typedef vector<uint, 4> uint4;
typedef vector<int32_t, 1> int32_t1;
typedef vector<int32_t, 2> int32_t2;
typedef vector<int32_t, 3> int32_t3;
typedef vector<int32_t, 4> int32_t4;
typedef vector<uint32_t, 1> uint32_t1;
typedef vector<uint32_t, 2> uint32_t2;
typedef vector<uint32_t, 3> uint32_t3;
typedef vector<uint32_t, 4> uint32_t4;
typedef vector<int64_t, 1> int64_t1;
typedef vector<int64_t, 2> int64_t2;
typedef vector<int64_t, 3> int64_t3;
typedef vector<int64_t, 4> int64_t4;
typedef vector<uint64_t, 1> uint64_t1;
typedef vector<uint64_t, 2> uint64_t2;
typedef vector<uint64_t, 3> uint64_t3;
typedef vector<uint64_t, 4> uint64_t4;

typedef vector<half, 1> half1;
typedef vector<half, 2> half2;
typedef vector<half, 3> half3;
typedef vector<half, 4> half4;
typedef vector<float, 1> float1;
typedef vector<float, 2> float2;
typedef vector<float, 3> float3;
typedef vector<float, 4> float4;
typedef vector<double, 1> double1;
typedef vector<double, 2> double2;
typedef vector<double, 3> double3;
typedef vector<double, 4> double4;

#ifdef __HLSL_ENABLE_16_BIT
typedef vector<float16_t, 1> float16_t1;
typedef vector<float16_t, 2> float16_t2;
typedef vector<float16_t, 3> float16_t3;
typedef vector<float16_t, 4> float16_t4;
#endif

typedef vector<float32_t, 1> float32_t1;
typedef vector<float32_t, 2> float32_t2;
typedef vector<float32_t, 3> float32_t3;
typedef vector<float32_t, 4> float32_t4;
typedef vector<float64_t, 1> float64_t1;
typedef vector<float64_t, 2> float64_t2;
typedef vector<float64_t, 3> float64_t3;
typedef vector<float64_t, 4> float64_t4;

#ifdef __HLSL_ENABLE_16_BIT
typedef matrix<int16_t, 1, 1> int16_t1x1;
typedef matrix<int16_t, 1, 2> int16_t1x2;
typedef matrix<int16_t, 1, 3> int16_t1x3;
typedef matrix<int16_t, 1, 4> int16_t1x4;
typedef matrix<int16_t, 2, 1> int16_t2x1;
typedef matrix<int16_t, 2, 2> int16_t2x2;
typedef matrix<int16_t, 2, 3> int16_t2x3;
typedef matrix<int16_t, 2, 4> int16_t2x4;
typedef matrix<int16_t, 3, 1> int16_t3x1;
typedef matrix<int16_t, 3, 2> int16_t3x2;
typedef matrix<int16_t, 3, 3> int16_t3x3;
typedef matrix<int16_t, 3, 4> int16_t3x4;
typedef matrix<int16_t, 4, 1> int16_t4x1;
typedef matrix<int16_t, 4, 2> int16_t4x2;
typedef matrix<int16_t, 4, 3> int16_t4x3;
typedef matrix<int16_t, 4, 4> int16_t4x4;
typedef matrix<uint16_t, 1, 1> uint16_t1x1;
typedef matrix<uint16_t, 1, 2> uint16_t1x2;
typedef matrix<uint16_t, 1, 3> uint16_t1x3;
typedef matrix<uint16_t, 1, 4> uint16_t1x4;
typedef matrix<uint16_t, 2, 1> uint16_t2x1;
typedef matrix<uint16_t, 2, 2> uint16_t2x2;
typedef matrix<uint16_t, 2, 3> uint16_t2x3;
typedef matrix<uint16_t, 2, 4> uint16_t2x4;
typedef matrix<uint16_t, 3, 1> uint16_t3x1;
typedef matrix<uint16_t, 3, 2> uint16_t3x2;
typedef matrix<uint16_t, 3, 3> uint16_t3x3;
typedef matrix<uint16_t, 3, 4> uint16_t3x4;
typedef matrix<uint16_t, 4, 1> uint16_t4x1;
typedef matrix<uint16_t, 4, 2> uint16_t4x2;
typedef matrix<uint16_t, 4, 3> uint16_t4x3;
typedef matrix<uint16_t, 4, 4> uint16_t4x4;
#endif

typedef matrix<int, 1, 1> int1x1;
typedef matrix<int, 1, 2> int1x2;
typedef matrix<int, 1, 3> int1x3;
typedef matrix<int, 1, 4> int1x4;
typedef matrix<int, 2, 1> int2x1;
typedef matrix<int, 2, 2> int2x2;
typedef matrix<int, 2, 3> int2x3;
typedef matrix<int, 2, 4> int2x4;
typedef matrix<int, 3, 1> int3x1;
typedef matrix<int, 3, 2> int3x2;
typedef matrix<int, 3, 3> int3x3;
typedef matrix<int, 3, 4> int3x4;
typedef matrix<int, 4, 1> int4x1;
typedef matrix<int, 4, 2> int4x2;
typedef matrix<int, 4, 3> int4x3;
typedef matrix<int, 4, 4> int4x4;
typedef matrix<uint, 1, 1> uint1x1;
typedef matrix<uint, 1, 2> uint1x2;
typedef matrix<uint, 1, 3> uint1x3;
typedef matrix<uint, 1, 4> uint1x4;
typedef matrix<uint, 2, 1> uint2x1;
typedef matrix<uint, 2, 2> uint2x2;
typedef matrix<uint, 2, 3> uint2x3;
typedef matrix<uint, 2, 4> uint2x4;
typedef matrix<uint, 3, 1> uint3x1;
typedef matrix<uint, 3, 2> uint3x2;
typedef matrix<uint, 3, 3> uint3x3;
typedef matrix<uint, 3, 4> uint3x4;
typedef matrix<uint, 4, 1> uint4x1;
typedef matrix<uint, 4, 2> uint4x2;
typedef matrix<uint, 4, 3> uint4x3;
typedef matrix<uint, 4, 4> uint4x4;
typedef matrix<int32_t, 1, 1> int32_t1x1;
typedef matrix<int32_t, 1, 2> int32_t1x2;
typedef matrix<int32_t, 1, 3> int32_t1x3;
typedef matrix<int32_t, 1, 4> int32_t1x4;
typedef matrix<int32_t, 2, 1> int32_t2x1;
typedef matrix<int32_t, 2, 2> int32_t2x2;
typedef matrix<int32_t, 2, 3> int32_t2x3;
typedef matrix<int32_t, 2, 4> int32_t2x4;
typedef matrix<int32_t, 3, 1> int32_t3x1;
typedef matrix<int32_t, 3, 2> int32_t3x2;
typedef matrix<int32_t, 3, 3> int32_t3x3;
typedef matrix<int32_t, 3, 4> int32_t3x4;
typedef matrix<int32_t, 4, 1> int32_t4x1;
typedef matrix<int32_t, 4, 2> int32_t4x2;
typedef matrix<int32_t, 4, 3> int32_t4x3;
typedef matrix<int32_t, 4, 4> int32_t4x4;
typedef matrix<uint32_t, 1, 1> uint32_t1x1;
typedef matrix<uint32_t, 1, 2> uint32_t1x2;
typedef matrix<uint32_t, 1, 3> uint32_t1x3;
typedef matrix<uint32_t, 1, 4> uint32_t1x4;
typedef matrix<uint32_t, 2, 1> uint32_t2x1;
typedef matrix<uint32_t, 2, 2> uint32_t2x2;
typedef matrix<uint32_t, 2, 3> uint32_t2x3;
typedef matrix<uint32_t, 2, 4> uint32_t2x4;
typedef matrix<uint32_t, 3, 1> uint32_t3x1;
typedef matrix<uint32_t, 3, 2> uint32_t3x2;
typedef matrix<uint32_t, 3, 3> uint32_t3x3;
typedef matrix<uint32_t, 3, 4> uint32_t3x4;
typedef matrix<uint32_t, 4, 1> uint32_t4x1;
typedef matrix<uint32_t, 4, 2> uint32_t4x2;
typedef matrix<uint32_t, 4, 3> uint32_t4x3;
typedef matrix<uint32_t, 4, 4> uint32_t4x4;
typedef matrix<int64_t, 1, 1> int64_t1x1;
typedef matrix<int64_t, 1, 2> int64_t1x2;
typedef matrix<int64_t, 1, 3> int64_t1x3;
typedef matrix<int64_t, 1, 4> int64_t1x4;
typedef matrix<int64_t, 2, 1> int64_t2x1;
typedef matrix<int64_t, 2, 2> int64_t2x2;
typedef matrix<int64_t, 2, 3> int64_t2x3;
typedef matrix<int64_t, 2, 4> int64_t2x4;
typedef matrix<int64_t, 3, 1> int64_t3x1;
typedef matrix<int64_t, 3, 2> int64_t3x2;
typedef matrix<int64_t, 3, 3> int64_t3x3;
typedef matrix<int64_t, 3, 4> int64_t3x4;
typedef matrix<int64_t, 4, 1> int64_t4x1;
typedef matrix<int64_t, 4, 2> int64_t4x2;
typedef matrix<int64_t, 4, 3> int64_t4x3;
typedef matrix<int64_t, 4, 4> int64_t4x4;
typedef matrix<uint64_t, 1, 1> uint64_t1x1;
typedef matrix<uint64_t, 1, 2> uint64_t1x2;
typedef matrix<uint64_t, 1, 3> uint64_t1x3;
typedef matrix<uint64_t, 1, 4> uint64_t1x4;
typedef matrix<uint64_t, 2, 1> uint64_t2x1;
typedef matrix<uint64_t, 2, 2> uint64_t2x2;
typedef matrix<uint64_t, 2, 3> uint64_t2x3;
typedef matrix<uint64_t, 2, 4> uint64_t2x4;
typedef matrix<uint64_t, 3, 1> uint64_t3x1;
typedef matrix<uint64_t, 3, 2> uint64_t3x2;
typedef matrix<uint64_t, 3, 3> uint64_t3x3;
typedef matrix<uint64_t, 3, 4> uint64_t3x4;
typedef matrix<uint64_t, 4, 1> uint64_t4x1;
typedef matrix<uint64_t, 4, 2> uint64_t4x2;
typedef matrix<uint64_t, 4, 3> uint64_t4x3;
typedef matrix<uint64_t, 4, 4> uint64_t4x4;

typedef matrix<half, 1, 1> half1x1;
typedef matrix<half, 1, 2> half1x2;
typedef matrix<half, 1, 3> half1x3;
typedef matrix<half, 1, 4> half1x4;
typedef matrix<half, 2, 1> half2x1;
typedef matrix<half, 2, 2> half2x2;
typedef matrix<half, 2, 3> half2x3;
typedef matrix<half, 2, 4> half2x4;
typedef matrix<half, 3, 1> half3x1;
typedef matrix<half, 3, 2> half3x2;
typedef matrix<half, 3, 3> half3x3;
typedef matrix<half, 3, 4> half3x4;
typedef matrix<half, 4, 1> half4x1;
typedef matrix<half, 4, 2> half4x2;
typedef matrix<half, 4, 3> half4x3;
typedef matrix<half, 4, 4> half4x4;
typedef matrix<float, 1, 1> float1x1;
typedef matrix<float, 1, 2> float1x2;
typedef matrix<float, 1, 3> float1x3;
typedef matrix<float, 1, 4> float1x4;
typedef matrix<float, 2, 1> float2x1;
typedef matrix<float, 2, 2> float2x2;
typedef matrix<float, 2, 3> float2x3;
typedef matrix<float, 2, 4> float2x4;
typedef matrix<float, 3, 1> float3x1;
typedef matrix<float, 3, 2> float3x2;
typedef matrix<float, 3, 3> float3x3;
typedef matrix<float, 3, 4> float3x4;
typedef matrix<float, 4, 1> float4x1;
typedef matrix<float, 4, 2> float4x2;
typedef matrix<float, 4, 3> float4x3;
typedef matrix<float, 4, 4> float4x4;
typedef matrix<double, 1, 1> double1x1;
typedef matrix<double, 1, 2> double1x2;
typedef matrix<double, 1, 3> double1x3;
typedef matrix<double, 1, 4> double1x4;
typedef matrix<double, 2, 1> double2x1;
typedef matrix<double, 2, 2> double2x2;
typedef matrix<double, 2, 3> double2x3;
typedef matrix<double, 2, 4> double2x4;
typedef matrix<double, 3, 1> double3x1;
typedef matrix<double, 3, 2> double3x2;
typedef matrix<double, 3, 3> double3x3;
typedef matrix<double, 3, 4> double3x4;
typedef matrix<double, 4, 1> double4x1;
typedef matrix<double, 4, 2> double4x2;
typedef matrix<double, 4, 3> double4x3;
typedef matrix<double, 4, 4> double4x4;

#ifdef __HLSL_ENABLE_16_BIT
typedef matrix<float16_t, 1, 1> float16_t1x1;
typedef matrix<float16_t, 1, 2> float16_t1x2;
typedef matrix<float16_t, 1, 3> float16_t1x3;
typedef matrix<float16_t, 1, 4> float16_t1x4;
typedef matrix<float16_t, 2, 1> float16_t2x1;
typedef matrix<float16_t, 2, 2> float16_t2x2;
typedef matrix<float16_t, 2, 3> float16_t2x3;
typedef matrix<float16_t, 2, 4> float16_t2x4;
typedef matrix<float16_t, 3, 1> float16_t3x1;
typedef matrix<float16_t, 3, 2> float16_t3x2;
typedef matrix<float16_t, 3, 3> float16_t3x3;
typedef matrix<float16_t, 3, 4> float16_t3x4;
typedef matrix<float16_t, 4, 1> float16_t4x1;
typedef matrix<float16_t, 4, 2> float16_t4x2;
typedef matrix<float16_t, 4, 3> float16_t4x3;
typedef matrix<float16_t, 4, 4> float16_t4x4;
#endif

typedef matrix<float32_t, 1, 1> float32_t1x1;
typedef matrix<float32_t, 1, 2> float32_t1x2;
typedef matrix<float32_t, 1, 3> float32_t1x3;
typedef matrix<float32_t, 1, 4> float32_t1x4;
typedef matrix<float32_t, 2, 1> float32_t2x1;
typedef matrix<float32_t, 2, 2> float32_t2x2;
typedef matrix<float32_t, 2, 3> float32_t2x3;
typedef matrix<float32_t, 2, 4> float32_t2x4;
typedef matrix<float32_t, 3, 1> float32_t3x1;
typedef matrix<float32_t, 3, 2> float32_t3x2;
typedef matrix<float32_t, 3, 3> float32_t3x3;
typedef matrix<float32_t, 3, 4> float32_t3x4;
typedef matrix<float32_t, 4, 1> float32_t4x1;
typedef matrix<float32_t, 4, 2> float32_t4x2;
typedef matrix<float32_t, 4, 3> float32_t4x3;
typedef matrix<float32_t, 4, 4> float32_t4x4;
typedef matrix<float64_t, 1, 1> float64_t1x1;
typedef matrix<float64_t, 1, 2> float64_t1x2;
typedef matrix<float64_t, 1, 3> float64_t1x3;
typedef matrix<float64_t, 1, 4> float64_t1x4;
typedef matrix<float64_t, 2, 1> float64_t2x1;
typedef matrix<float64_t, 2, 2> float64_t2x2;
typedef matrix<float64_t, 2, 3> float64_t2x3;
typedef matrix<float64_t, 2, 4> float64_t2x4;
typedef matrix<float64_t, 3, 1> float64_t3x1;
typedef matrix<float64_t, 3, 2> float64_t3x2;
typedef matrix<float64_t, 3, 3> float64_t3x3;
typedef matrix<float64_t, 3, 4> float64_t3x4;
typedef matrix<float64_t, 4, 1> float64_t4x1;
typedef matrix<float64_t, 4, 2> float64_t4x2;
typedef matrix<float64_t, 4, 3> float64_t4x3;
typedef matrix<float64_t, 4, 4> float64_t4x4;

} // namespace hlsl

#endif //_HLSL_HLSL_BASIC_TYPES_H_
