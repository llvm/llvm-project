//===----- hlsl_basic_types.h - HLSL definitions for basic types ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_BASIC_TYPES_H_
#define _HLSL_HLSL_BASIC_TYPES_H_

// built-in scalar data types:

#ifdef __HLSL_ENABLE_16_BIT
// 16-bit integer.
typedef unsigned short uint16_t;
typedef short int16_t;
#endif

// unsigned 32-bit integer.
typedef unsigned int uint;

// 64-bit integer.
typedef unsigned long uint64_t;
typedef long int64_t;

// built-in vector data types:

#ifdef __HLSL_ENABLE_16_BIT
typedef int16_t int16_t2 __attribute__((ext_vector_type(2)));
typedef int16_t int16_t3 __attribute__((ext_vector_type(3)));
typedef int16_t int16_t4 __attribute__((ext_vector_type(4)));
typedef uint16_t uint16_t2 __attribute__((ext_vector_type(2)));
typedef uint16_t uint16_t3 __attribute__((ext_vector_type(3)));
typedef uint16_t uint16_t4 __attribute__((ext_vector_type(4)));
#endif

typedef int int2 __attribute__((ext_vector_type(2)));
typedef int int3 __attribute__((ext_vector_type(3)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef uint uint2 __attribute__((ext_vector_type(2)));
typedef uint uint3 __attribute__((ext_vector_type(3)));
typedef uint uint4 __attribute__((ext_vector_type(4)));
typedef int64_t int64_t2 __attribute__((ext_vector_type(2)));
typedef int64_t int64_t3 __attribute__((ext_vector_type(3)));
typedef int64_t int64_t4 __attribute__((ext_vector_type(4)));
typedef uint64_t uint64_t2 __attribute__((ext_vector_type(2)));
typedef uint64_t uint64_t3 __attribute__((ext_vector_type(3)));
typedef uint64_t uint64_t4 __attribute__((ext_vector_type(4)));

#ifdef __HLSL_ENABLE_16_BIT
typedef half half2 __attribute__((ext_vector_type(2)));
typedef half half3 __attribute__((ext_vector_type(3)));
typedef half half4 __attribute__((ext_vector_type(4)));
#endif

typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef double double2 __attribute__((ext_vector_type(2)));
typedef double double3 __attribute__((ext_vector_type(3)));
typedef double double4 __attribute__((ext_vector_type(4)));

#endif //_HLSL_HLSL_BASIC_TYPES_H_
