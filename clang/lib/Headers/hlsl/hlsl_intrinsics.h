//===----- hlsl_intrinsics.h - HLSL definitions for intrinsics ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_INTRINSICS_H_
#define _HLSL_HLSL_INTRINSICS_H_

namespace hlsl {

// Note: Functions in this file are sorted alphabetically, then grouped by base
// element type, and the element types are sorted by size, then singed integer,
// unsigned integer and floating point. Keeping this ordering consistent will
// help keep this file manageable as it grows.

#define _HLSL_BUILTIN_ALIAS(builtin)                                           \
  __attribute__((clang_builtin_alias(builtin)))
#define _HLSL_AVAILABILITY(environment, version)                               \
  __attribute__((availability(environment, introduced = version)))

#ifdef __HLSL_ENABLE_16_BIT
#define _HLSL_16BIT_AVAILABILITY(environment, version)                         \
  __attribute__((availability(environment, introduced = version)))
#else
#define _HLSL_16BIT_AVAILABILITY(environment, version)
#endif

//===----------------------------------------------------------------------===//
// abs builtins
//===----------------------------------------------------------------------===//

/// \fn T abs(T Val)
/// \brief Returns the absolute value of the input value, \a Val.
/// \param Val The input value.

#ifdef __HLSL_ENABLE_16_BIT
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
int16_t abs(int16_t);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
int16_t2 abs(int16_t2);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
int16_t3 abs(int16_t3);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
int16_t4 abs(int16_t4);
#endif

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
half abs(half);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
half2 abs(half2);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
half3 abs(half3);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
half4 abs(half4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
int abs(int);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
int2 abs(int2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
int3 abs(int3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
int4 abs(int4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
float abs(float);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
float2 abs(float2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
float3 abs(float3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
float4 abs(float4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
int64_t abs(int64_t);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
int64_t2 abs(int64_t2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
int64_t3 abs(int64_t3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
int64_t4 abs(int64_t4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
double abs(double);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
double2 abs(double2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
double3 abs(double3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_abs)
double4 abs(double4);

//===----------------------------------------------------------------------===//
// ceil builtins
//===----------------------------------------------------------------------===//

/// \fn T ceil(T Val)
/// \brief Returns the smallest integer value that is greater than or equal to
/// the input value, \a Val.
/// \param Val The input value.

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_ceil)
half ceil(half);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_ceil)
half2 ceil(half2);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_ceil)
half3 ceil(half3);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_ceil)
half4 ceil(half4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_ceil)
float ceil(float);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_ceil)
float2 ceil(float2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_ceil)
float3 ceil(float3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_ceil)
float4 ceil(float4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_ceil)
double ceil(double);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_ceil)
double2 ceil(double2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_ceil)
double3 ceil(double3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_ceil)
double4 ceil(double4);

//===----------------------------------------------------------------------===//
// cos builtins
//===----------------------------------------------------------------------===//

/// \fn T cos(T Val)
/// \brief Returns the cosine of the input value, \a Val.
/// \param Val The input value.

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_cos)
half cos(half);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_cos)
half2 cos(half2);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_cos)
half3 cos(half3);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_cos)
half4 cos(half4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_cos)
float cos(float);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_cos)
float2 cos(float2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_cos)
float3 cos(float3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_cos)
float4 cos(float4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_cos)
double cos(double);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_cos)
double2 cos(double2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_cos)
double3 cos(double3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_cos)
double4 cos(double4);

//===----------------------------------------------------------------------===//
// dot product builtins
//===----------------------------------------------------------------------===//

/// \fn K dot(T X, T Y)
/// \brief Return the dot product (a scalar value) of \a X and \a Y.
/// \param X The X input value.
/// \param Y The Y input value.

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
half dot(half, half);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
half dot(half2, half2);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
half dot(half3, half3);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
half dot(half4, half4);

#ifdef __HLSL_ENABLE_16_BIT
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
int16_t dot(int16_t, int16_t);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
int16_t dot(int16_t2, int16_t2);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
int16_t dot(int16_t3, int16_t3);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
int16_t dot(int16_t4, int16_t4);

_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
uint16_t dot(uint16_t, uint16_t);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
uint16_t dot(uint16_t2, uint16_t2);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
uint16_t dot(uint16_t3, uint16_t3);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
uint16_t dot(uint16_t4, uint16_t4);
#endif

_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
float dot(float, float);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
float dot(float2, float2);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
float dot(float3, float3);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
float dot(float4, float4);

_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
double dot(double, double);

_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
int dot(int, int);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
int dot(int2, int2);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
int dot(int3, int3);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
int dot(int4, int4);

_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
uint dot(uint, uint);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
uint dot(uint2, uint2);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
uint dot(uint3, uint3);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
uint dot(uint4, uint4);

_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
int64_t dot(int64_t, int64_t);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
int64_t dot(int64_t2, int64_t2);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
int64_t dot(int64_t3, int64_t3);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
int64_t dot(int64_t4, int64_t4);

_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
uint64_t dot(uint64_t, uint64_t);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
uint64_t dot(uint64_t2, uint64_t2);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
uint64_t dot(uint64_t3, uint64_t3);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot)
uint64_t dot(uint64_t4, uint64_t4);

//===----------------------------------------------------------------------===//
// floor builtins
//===----------------------------------------------------------------------===//

/// \fn T floor(T Val)
/// \brief Returns the largest integer that is less than or equal to the input
/// value, \a Val.
/// \param Val The input value.

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_floor)
half floor(half);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_floor)
half2 floor(half2);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_floor)
half3 floor(half3);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_floor)
half4 floor(half4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_floor)
float floor(float);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_floor)
float2 floor(float2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_floor)
float3 floor(float3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_floor)
float4 floor(float4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_floor)
double floor(double);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_floor)
double2 floor(double2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_floor)
double3 floor(double3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_floor)
double4 floor(double4);

//===----------------------------------------------------------------------===//
// log builtins
//===----------------------------------------------------------------------===//

/// \fn T log(T Val)
/// \brief The base-e logarithm of the input value, \a Val parameter.
/// \param Val The input value.
///
/// If \a Val is negative, this result is undefined. If \a Val is 0, this
/// function returns negative infinity.

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log)
half log(half);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log)
half2 log(half2);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log)
half3 log(half3);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log)
half4 log(half4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log)
float log(float);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log)
float2 log(float2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log)
float3 log(float3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log)
float4 log(float4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log)
double log(double);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log)
double2 log(double2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log)
double3 log(double3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log)
double4 log(double4);

//===----------------------------------------------------------------------===//
// log10 builtins
//===----------------------------------------------------------------------===//

/// \fn T log10(T Val)
/// \brief The base-10 logarithm of the input value, \a Val parameter.
/// \param Val The input value.
///
/// If \a Val is negative, this result is undefined. If \a Val is 0, this
/// function returns negative infinity.

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log10)
half log10(half);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log10)
half2 log10(half2);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log10)
half3 log10(half3);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log10)
half4 log10(half4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log10)
float log10(float);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log10)
float2 log10(float2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log10)
float3 log10(float3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log10)
float4 log10(float4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log10)
double log10(double);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log10)
double2 log10(double2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log10)
double3 log10(double3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log10)
double4 log10(double4);

//===----------------------------------------------------------------------===//
// log2 builtins
//===----------------------------------------------------------------------===//

/// \fn T log2(T Val)
/// \brief The base-2 logarithm of the input value, \a Val parameter.
/// \param Val The input value.
///
/// If \a Val is negative, this result is undefined. If \a Val is 0, this
/// function returns negative infinity.

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log2)
half log2(half);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log2)
half2 log2(half2);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log2)
half3 log2(half3);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log2)
half4 log2(half4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log2)
float log2(float);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log2)
float2 log2(float2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log2)
float3 log2(float3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log2)
float4 log2(float4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log2)
double log2(double);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log2)
double2 log2(double2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log2)
double3 log2(double3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_log2)
double4 log2(double4);

//===----------------------------------------------------------------------===//
// max builtins
//===----------------------------------------------------------------------===//

/// \fn T max(T X, T Y)
/// \brief Return the greater of \a X and \a Y.
/// \param X The X input value.
/// \param Y The Y input value.

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
half max(half, half);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
half2 max(half2, half2);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
half3 max(half3, half3);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
half4 max(half4, half4);

#ifdef __HLSL_ENABLE_16_BIT
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
int16_t max(int16_t, int16_t);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
int16_t2 max(int16_t2, int16_t2);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
int16_t3 max(int16_t3, int16_t3);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
int16_t4 max(int16_t4, int16_t4);

_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
uint16_t max(uint16_t, uint16_t);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
uint16_t2 max(uint16_t2, uint16_t2);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
uint16_t3 max(uint16_t3, uint16_t3);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
uint16_t4 max(uint16_t4, uint16_t4);
#endif

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
int max(int, int);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
int2 max(int2, int2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
int3 max(int3, int3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
int4 max(int4, int4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
uint max(uint, uint);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
uint2 max(uint2, uint2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
uint3 max(uint3, uint3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
uint4 max(uint4, uint4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
int64_t max(int64_t, int64_t);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
int64_t2 max(int64_t2, int64_t2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
int64_t3 max(int64_t3, int64_t3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
int64_t4 max(int64_t4, int64_t4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
uint64_t max(uint64_t, uint64_t);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
uint64_t2 max(uint64_t2, uint64_t2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
uint64_t3 max(uint64_t3, uint64_t3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
uint64_t4 max(uint64_t4, uint64_t4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
float max(float, float);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
float2 max(float2, float2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
float3 max(float3, float3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
float4 max(float4, float4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
double max(double, double);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
double2 max(double2, double2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
double3 max(double3, double3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_max)
double4 max(double4, double4);

//===----------------------------------------------------------------------===//
// min builtins
//===----------------------------------------------------------------------===//

/// \fn T min(T X, T Y)
/// \brief Return the lesser of \a X and \a Y.
/// \param X The X input value.
/// \param Y The Y input value.

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
half min(half, half);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
half2 min(half2, half2);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
half3 min(half3, half3);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
half4 min(half4, half4);

#ifdef __HLSL_ENABLE_16_BIT
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
int16_t min(int16_t, int16_t);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
int16_t2 min(int16_t2, int16_t2);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
int16_t3 min(int16_t3, int16_t3);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
int16_t4 min(int16_t4, int16_t4);

_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
uint16_t min(uint16_t, uint16_t);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
uint16_t2 min(uint16_t2, uint16_t2);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
uint16_t3 min(uint16_t3, uint16_t3);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
uint16_t4 min(uint16_t4, uint16_t4);
#endif

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
int min(int, int);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
int2 min(int2, int2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
int3 min(int3, int3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
int4 min(int4, int4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
uint min(uint, uint);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
uint2 min(uint2, uint2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
uint3 min(uint3, uint3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
uint4 min(uint4, uint4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
float min(float, float);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
float2 min(float2, float2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
float3 min(float3, float3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
float4 min(float4, float4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
int64_t min(int64_t, int64_t);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
int64_t2 min(int64_t2, int64_t2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
int64_t3 min(int64_t3, int64_t3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
int64_t4 min(int64_t4, int64_t4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
uint64_t min(uint64_t, uint64_t);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
uint64_t2 min(uint64_t2, uint64_t2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
uint64_t3 min(uint64_t3, uint64_t3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
uint64_t4 min(uint64_t4, uint64_t4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
double min(double, double);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
double2 min(double2, double2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
double3 min(double3, double3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_min)
double4 min(double4, double4);

//===----------------------------------------------------------------------===//
// pow builtins
//===----------------------------------------------------------------------===//

/// \fn T pow(T Val, T Pow)
/// \brief Return the value \a Val, raised to the power \a Pow.
/// \param Val The input value.
/// \param Pow The specified power.

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_pow)
half pow(half, half);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_pow)
half2 pow(half2, half2);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_pow)
half3 pow(half3, half3);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_pow)
half4 pow(half4, half4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_pow)
float pow(float, float);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_pow)
float2 pow(float2, float2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_pow)
float3 pow(float3, float3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_pow)
float4 pow(float4, float4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_pow)
double pow(double, double);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_pow)
double2 pow(double2, double2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_pow)
double3 pow(double3, double3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_pow)
double4 pow(double4, double4);

//===----------------------------------------------------------------------===//
// reversebits builtins
//===----------------------------------------------------------------------===//

/// \fn T reversebits(T Val)
/// \brief Return the value \a Val with the bit order reversed.
/// \param Val The input value.

#ifdef __HLSL_ENABLE_16_BIT
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
int16_t reversebits(int16_t);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
int16_t2 reversebits(int16_t2);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
int16_t3 reversebits(int16_t3);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
int16_t4 reversebits(int16_t4);

_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
uint16_t reversebits(uint16_t);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
uint16_t2 reversebits(uint16_t2);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
uint16_t3 reversebits(uint16_t3);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
uint16_t4 reversebits(uint16_t4);
#endif

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
int reversebits(int);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
int2 reversebits(int2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
int3 reversebits(int3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
int4 reversebits(int4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
uint reversebits(uint);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
uint2 reversebits(uint2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
uint3 reversebits(uint3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
uint4 reversebits(uint4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
int64_t reversebits(int64_t);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
int64_t2 reversebits(int64_t2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
int64_t3 reversebits(int64_t3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
int64_t4 reversebits(int64_t4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
uint64_t reversebits(uint64_t);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
uint64_t2 reversebits(uint64_t2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
uint64_t3 reversebits(uint64_t3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_bitreverse)
uint64_t4 reversebits(uint64_t4);

//===----------------------------------------------------------------------===//
// sin builtins
//===----------------------------------------------------------------------===//

/// \fn T sin(T Val)
/// \brief Returns the sine of the input value, \a Val.
/// \param Val The input value.

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_sin)
half sin(half);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_sin)
half2 sin(half2);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_sin)
half3 sin(half3);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_sin)
half4 sin(half4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_sin)
float sin(float);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_sin)
float2 sin(float2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_sin)
float3 sin(float3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_sin)
float4 sin(float4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_sin)
double sin(double);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_sin)
double2 sin(double2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_sin)
double3 sin(double3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_sin)
double4 sin(double4);

//===----------------------------------------------------------------------===//
// sqrt builtins
//===----------------------------------------------------------------------===//

/// \fn T sqrt(T Val)
/// \brief Returns the square root of the input value, \a Val.
/// \param Val The input value.

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_sqrtf16)
half sqrt(half In);

_HLSL_BUILTIN_ALIAS(__builtin_sqrtf)
float sqrt(float In);

_HLSL_BUILTIN_ALIAS(__builtin_sqrt)
double sqrt(double In);

//===----------------------------------------------------------------------===//
// trunc builtins
//===----------------------------------------------------------------------===//

/// \fn T trunc(T Val)
/// \brief Returns the truncated integer value of the input value, \a Val.
/// \param Val The input value.

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_trunc)
half trunc(half);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_trunc)
half2 trunc(half2);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_trunc)
half3 trunc(half3);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_trunc)
half4 trunc(half4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_trunc)
float trunc(float);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_trunc)
float2 trunc(float2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_trunc)
float3 trunc(float3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_trunc)
float4 trunc(float4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_trunc)
double trunc(double);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_trunc)
double2 trunc(double2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_trunc)
double3 trunc(double3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_trunc)
double4 trunc(double4);

//===----------------------------------------------------------------------===//
// Wave* builtins
//===----------------------------------------------------------------------===//

/// \brief Counts the number of boolean variables which evaluate to true across
/// all active lanes in the current wave.
///
/// \param Val The input boolean value.
/// \return The number of lanes for which the boolean variable evaluates to
/// true, across all active lanes in the current wave.
_HLSL_AVAILABILITY(shadermodel, 6.0)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_wave_active_count_bits)
uint WaveActiveCountBits(bool Val);

} // namespace hlsl
#endif //_HLSL_HLSL_INTRINSICS_H_
