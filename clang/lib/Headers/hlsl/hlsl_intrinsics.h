//===----- hlsl_intrinsics.h - HLSL definitions for intrinsics ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_INTRINSICS_H_
#define _HLSL_HLSL_INTRINSICS_H_

__attribute__((availability(shadermodel, introduced = 6.0)))
__attribute__((clang_builtin_alias(__builtin_hlsl_wave_active_count_bits))) uint
WaveActiveCountBits(bool bBit);

#ifdef __HLSL_ENABLE_16_BIT
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
int16_t abs(int16_t);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
int16_t2 abs(int16_t2);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
int16_t3 abs(int16_t3);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
int16_t4 abs(int16_t4);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs))) half abs(half);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
half2 abs(half2);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
half3 abs(half3);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
half4 abs(half4);
#endif

__attribute__((clang_builtin_alias(__builtin_elementwise_abs))) int abs(int);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs))) int2 abs(int2);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs))) int3 abs(int3);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs))) int4 abs(int4);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs))) float
abs(float);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
float2 abs(float2);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
float3 abs(float3);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
float4 abs(float4);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
int64_t abs(int64_t);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
int64_t2 abs(int64_t2);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
int64_t3 abs(int64_t3);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
int64_t4 abs(int64_t4);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs))) double
abs(double);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
double2 abs(double2);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
double3 abs(double3);
__attribute__((clang_builtin_alias(__builtin_elementwise_abs)))
double4 abs(double4);

#endif //_HLSL_HLSL_INTRINSICS_H_
