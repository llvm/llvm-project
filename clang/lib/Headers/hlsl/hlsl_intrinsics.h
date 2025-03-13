//===----- hlsl_intrinsics.h - HLSL definitions for intrinsics ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_INTRINSICS_H_
#define _HLSL_HLSL_INTRINSICS_H_

#include "hlsl/hlsl_intrinsic_helpers.h"

namespace hlsl {

//===----------------------------------------------------------------------===//
// asfloat builtins
//===----------------------------------------------------------------------===//

/// \fn float asfloat(T Val)
/// \brief Interprets the bit pattern of x as float point number.
/// \param Val The input value.

template <typename T, int N>
constexpr vector<float, N> asfloat(vector<T, N> V) {
  return __detail::bit_cast<float, T, N>(V);
}

template <typename T> constexpr float asfloat(T F) {
  return __detail::bit_cast<float, T>(F);
}

//===----------------------------------------------------------------------===//
// asint builtins
//===----------------------------------------------------------------------===//

/// \fn int asint(T Val)
/// \brief Interprets the bit pattern of x as an integer.
/// \param Val The input value.

template <typename T, int N> constexpr vector<int, N> asint(vector<T, N> V) {
  return __detail::bit_cast<int, T, N>(V);
}

template <typename T> constexpr int asint(T F) {
  return __detail::bit_cast<int, T>(F);
}

//===----------------------------------------------------------------------===//
// asuint builtins
//===----------------------------------------------------------------------===//

/// \fn uint asuint(T Val)
/// \brief Interprets the bit pattern of x as an unsigned integer.
/// \param Val The input value.

template <typename T, int N> constexpr vector<uint, N> asuint(vector<T, N> V) {
  return __detail::bit_cast<uint, T, N>(V);
}

template <typename T> constexpr uint asuint(T F) {
  return __detail::bit_cast<uint, T>(F);
}

//===----------------------------------------------------------------------===//
// asuint splitdouble builtins
//===----------------------------------------------------------------------===//

/// \fn void asuint(double D, out uint lowbits, out int highbits)
/// \brief Split and interprets the lowbits and highbits of double D into uints.
/// \param D The input double.
/// \param lowbits The output lowbits of D.
/// \param highbits The output highbits of D.
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_splitdouble)
void asuint(double, out uint, out uint);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_splitdouble)
void asuint(double2, out uint2, out uint2);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_splitdouble)
void asuint(double3, out uint3, out uint3);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_splitdouble)
void asuint(double4, out uint4, out uint4);

//===----------------------------------------------------------------------===//
// distance builtins
//===----------------------------------------------------------------------===//

/// \fn K distance(T X, T Y)
/// \brief Returns a distance scalar between \a X and \a Y.
/// \param X The X input value.
/// \param Y The Y input value.

template <typename T>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::enable_if_t<__detail::is_arithmetic<T>::Value &&
                                       __detail::is_same<half, T>::value,
                                   T> distance(T X, T Y) {
  return __detail::distance_impl(X, Y);
}

template <typename T>
const inline __detail::enable_if_t<
    __detail::is_arithmetic<T>::Value && __detail::is_same<float, T>::value, T>
distance(T X, T Y) {
  return __detail::distance_impl(X, Y);
}

template <int N>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline half distance(__detail::HLSL_FIXED_VECTOR<half, N> X,
                           __detail::HLSL_FIXED_VECTOR<half, N> Y) {
  return __detail::distance_vec_impl(X, Y);
}

template <int N>
const inline float distance(__detail::HLSL_FIXED_VECTOR<float, N> X,
                            __detail::HLSL_FIXED_VECTOR<float, N> Y) {
  return __detail::distance_vec_impl(X, Y);
}

//===----------------------------------------------------------------------===//
<<<<<<< Updated upstream
=======
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
// dot4add builtins
//===----------------------------------------------------------------------===//

/// \fn int dot4add_i8packed(uint A, uint B, int C)

_HLSL_AVAILABILITY(shadermodel, 6.4)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot4add_i8packed)
int dot4add_i8packed(uint, uint, int);

/// \fn uint dot4add_u8packed(uint A, uint B, uint C)

_HLSL_AVAILABILITY(shadermodel, 6.4)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot4add_u8packed)
uint dot4add_u8packed(uint, uint, uint);

//===----------------------------------------------------------------------===//
// dot2add builtins
//===----------------------------------------------------------------------===//

/// \fn float dot2add(half2 a, half2 b, float c)

_HLSL_AVAILABILITY(shadermodel, 6.4)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_dot2add)
float dot2add(half2, half2, float);

//===----------------------------------------------------------------------===//
// exp builtins
//===----------------------------------------------------------------------===//

/// \fn T exp(T x)
/// \brief Returns the base-e exponential, or \a e**x, of the specified value.
/// \param x The specified input value.
///
/// The return value is the base-e exponential of the \a x parameter.

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_exp)
half exp(half);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_exp)
half2 exp(half2);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_exp)
half3 exp(half3);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_exp)
half4 exp(half4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_exp)
float exp(float);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_exp)
float2 exp(float2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_exp)
float3 exp(float3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_exp)
float4 exp(float4);

//===----------------------------------------------------------------------===//
// exp2 builtins
//===----------------------------------------------------------------------===//

/// \fn T exp2(T x)
/// \brief Returns the base 2 exponential, or \a 2**x, of the specified value.
/// \param x The specified input value.
///
/// The base 2 exponential of the \a x parameter.

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_exp2)
half exp2(half);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_exp2)
half2 exp2(half2);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_exp2)
half3 exp2(half3);
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_exp2)
half4 exp2(half4);

_HLSL_BUILTIN_ALIAS(__builtin_elementwise_exp2)
float exp2(float);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_exp2)
float2 exp2(float2);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_exp2)
float3 exp2(float3);
_HLSL_BUILTIN_ALIAS(__builtin_elementwise_exp2)
float4 exp2(float4);

//===----------------------------------------------------------------------===//
// firstbithigh builtins
//===----------------------------------------------------------------------===//

/// \fn T firstbithigh(T Val)
/// \brief Returns the location of the first set bit starting from the highest
/// order bit and working downward, per component.
/// \param Val the input value.

#ifdef __HLSL_ENABLE_16_BIT
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint firstbithigh(int16_t);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint2 firstbithigh(int16_t2);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint3 firstbithigh(int16_t3);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint4 firstbithigh(int16_t4);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint firstbithigh(uint16_t);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint2 firstbithigh(uint16_t2);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint3 firstbithigh(uint16_t3);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint4 firstbithigh(uint16_t4);
#endif

_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint firstbithigh(int);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint2 firstbithigh(int2);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint3 firstbithigh(int3);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint4 firstbithigh(int4);

_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint firstbithigh(uint);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint2 firstbithigh(uint2);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint3 firstbithigh(uint3);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint4 firstbithigh(uint4);

_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint firstbithigh(int64_t);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint2 firstbithigh(int64_t2);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint3 firstbithigh(int64_t3);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint4 firstbithigh(int64_t4);

_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint firstbithigh(uint64_t);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint2 firstbithigh(uint64_t2);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint3 firstbithigh(uint64_t3);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbithigh)
uint4 firstbithigh(uint64_t4);

//===----------------------------------------------------------------------===//
// firstbitlow builtins
//===----------------------------------------------------------------------===//

/// \fn T firstbitlow(T Val)
/// \brief Returns the location of the first set bit starting from the lowest
/// order bit and working upward, per component.
/// \param Val the input value.

#ifdef __HLSL_ENABLE_16_BIT
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint firstbitlow(int16_t);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint2 firstbitlow(int16_t2);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint3 firstbitlow(int16_t3);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint4 firstbitlow(int16_t4);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint firstbitlow(uint16_t);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint2 firstbitlow(uint16_t2);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint3 firstbitlow(uint16_t3);
_HLSL_AVAILABILITY(shadermodel, 6.2)
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint4 firstbitlow(uint16_t4);
#endif

_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint firstbitlow(int);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint2 firstbitlow(int2);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint3 firstbitlow(int3);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint4 firstbitlow(int4);

_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint firstbitlow(uint);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint2 firstbitlow(uint2);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint3 firstbitlow(uint3);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint4 firstbitlow(uint4);

_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint firstbitlow(int64_t);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint2 firstbitlow(int64_t2);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint3 firstbitlow(int64_t3);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint4 firstbitlow(int64_t4);

_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint firstbitlow(uint64_t);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint2 firstbitlow(uint64_t2);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint3 firstbitlow(uint64_t3);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_firstbitlow)
uint4 firstbitlow(uint64_t4);

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

//===----------------------------------------------------------------------===//
>>>>>>> Stashed changes
// fmod builtins
//===----------------------------------------------------------------------===//

/// \fn T fmod(T x, T y)
/// \brief Returns the linear interpolation of x to y.
/// \param x [in] The dividend.
/// \param y [in] The divisor.
///
/// Return the floating-point remainder of the x parameter divided by the y
/// parameter.

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline half fmod(half X, half Y) { return __detail::fmod_impl(X, Y); }

const inline float fmod(float X, float Y) { return __detail::fmod_impl(X, Y); }

template <int N>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline vector<half, N> fmod(vector<half, N> X, vector<half, N> Y) {
  return __detail::fmod_vec_impl(X, Y);
}

template <int N>
const inline vector<float, N> fmod(vector<float, N> X, vector<float, N> Y) {
  return __detail::fmod_vec_impl(X, Y);
}

//===----------------------------------------------------------------------===//
// length builtins
//===----------------------------------------------------------------------===//

/// \fn T length(T x)
/// \brief Returns the length of the specified floating-point vector.
/// \param x [in] The vector of floats, or a scalar float.
///
/// Length is based on the following formula: sqrt(x[0]^2 + x[1]^2 + ...).

template <typename T>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::enable_if_t<__detail::is_arithmetic<T>::Value &&
                                       __detail::is_same<half, T>::value,
                                   T> length(T X) {
  return __detail::length_impl(X);
}

template <typename T>
const inline __detail::enable_if_t<
    __detail::is_arithmetic<T>::Value && __detail::is_same<float, T>::value, T>
length(T X) {
  return __detail::length_impl(X);
}

template <int N>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline half length(__detail::HLSL_FIXED_VECTOR<half, N> X) {
  return __detail::length_vec_impl(X);
}

template <int N>
const inline float length(__detail::HLSL_FIXED_VECTOR<float, N> X) {
  return __detail::length_vec_impl(X);
}

//===----------------------------------------------------------------------===//
// D3DCOLORtoUBYTE4 builtin
//===----------------------------------------------------------------------===//

/// \fn T D3DCOLORtoUBYTE4(T x)
/// \brief Converts a floating-point, 4D vector set by a D3DCOLOR to a UBYTE4.
/// \param x [in] The floating-point vector4 to convert.
///
/// The return value is the UBYTE4 representation of the \a x parameter.
///
/// This function swizzles and scales components of the \a x parameter. Use this
/// function to compensate for the lack of UBYTE4 support in some hardware.

constexpr vector<uint, 4> D3DCOLORtoUBYTE4(vector<float, 4> V) {
  return __detail::d3d_color_to_ubyte4_impl(V);
}

//===----------------------------------------------------------------------===//
// reflect builtin
//===----------------------------------------------------------------------===//

/// \fn T reflect(T I, T N)
/// \brief Returns a reflection using an incident ray, \a I, and a surface
/// normal, \a N.
/// \param I The incident ray.
/// \param N The surface normal.
///
/// The return value is a floating-point vector that represents the reflection
/// of the incident ray, \a I, off a surface with the normal \a N.
///
/// This function calculates the reflection vector using the following formula:
/// V = I - 2 * N * dot(I N) .
///
/// N must already be normalized in order to achieve the desired result.
///
/// The operands must all be a scalar or vector whose component type is
/// floating-point.
///
/// Result type and the type of all operands must be the same type.

template <typename T>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::enable_if_t<__detail::is_arithmetic<T>::Value &&
                                       __detail::is_same<half, T>::value,
                                   T> reflect(T I, T N) {
  return __detail::reflect_impl(I, N);
}

template <typename T>
const inline __detail::enable_if_t<
    __detail::is_arithmetic<T>::Value && __detail::is_same<float, T>::value, T>
reflect(T I, T N) {
  return __detail::reflect_impl(I, N);
}

template <int L>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::HLSL_FIXED_VECTOR<half, L> reflect(
    __detail::HLSL_FIXED_VECTOR<half, L> I,
    __detail::HLSL_FIXED_VECTOR<half, L> N) {
  return __detail::reflect_vec_impl(I, N);
}

template <int L>
const inline __detail::HLSL_FIXED_VECTOR<float, L>
reflect(__detail::HLSL_FIXED_VECTOR<float, L> I,
        __detail::HLSL_FIXED_VECTOR<float, L> N) {
  return __detail::reflect_vec_impl(I, N);
}
} // namespace hlsl
#endif //_HLSL_HLSL_INTRINSICS_H_
